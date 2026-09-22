# Clouds

A research renderer for real-time, high-quality volumetric clouds at 4K.

The performance target is **below 2 ms at 3840×2160** while preserving the required image quality against the saved path-traced reference.

## Licensing

HST is derived from NVIDIA Falcor. HST-specific original work authored by Sam Frisby is licensed under Apache-2.0. Falcor-derived portions remain subject to NVIDIA's BSD 3-Clause license, and bundled third-party components retain their own licenses. See [LICENSE.md](LICENSE.md), [LICENSES](LICENSES), and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

This project is pursuing a **radical camera-rendering architecture**. It is **not** trying to make conventional volumetric ray marching incrementally faster.

For bounded, headless GPU/CPU timeline captures, see [Profiling HSTR with NVIDIA Nsight Systems](docs/HSTR-nsight-systems.md).

---

# Read This Before Changing the Renderer

## The final renderer is NOT a ray marcher

The fundamental design goal is:

> **Camera rays must stop being the primary unit of rendering work.**

The final architecture must not render the image by selecting some set of screen positions and then running an expensive density march for each one.

This includes seemingly clever variants such as:

```text
adaptive tiles
    ↓
sparse sample positions
    ↓
marchBeam()
    ↓
interpolate
```

That is still a ray marcher.

It merely ray marches fewer pixels.

It is **not the architecture this project is trying to build**.

---

# Final Architecture

The renderer should operate primarily on:

```text
projected transport regions
×
screen-space beam tiles
```

rather than:

```text
camera rays
×
volume steps
```

The intended dataflow is:

```text
                    WORLD SPACE

cloud density / virtual residency
             ↓
world-space transport representation
             ↓
adaptive HSTR transport cut
             ↓
accepted nodes carrying compact transport information
             ↓
high-frequency information represented separately as residuals


                    CAMERA SPACE

accepted transport cut
             ↓
project nodes into screen space
             ↓
bin nodes into adaptive BeamTiles
             ↓
front-to-back tile/node work
             ↓
evaluate node transfer directly into tile coefficients
             ↓
adaptive BeamTile subdivision where required
             ↓
compact residual work queue for genuinely unresolved detail
             ↓
BeamTile coefficient field
             ↓
cheap full-resolution reconstruction
             ↓
3840×2160 output
```

The critical transformation is:

```text
OLD

screen sample
    ↓
walk through volume
    ↓
sample density repeatedly
    ↓
sample lighting repeatedly
    ↓
accumulate radiance
```

becoming:

```text
FINAL

projected transport node
       ×
screen BeamTile
       ↓
evaluate compact transfer representation
       ↓
accumulate tile coefficients
```

The ordinary camera path should therefore have **no conventional volume march**.

---

# What “Beam” Means

Beam is not just an interpolation layer placed after ray marching.

Beam is intended to become the **camera-space rendering representation itself**.

A `BeamTile` represents the radiometric behaviour of a screen region.

Conceptually it contains coefficients sufficient to reconstruct quantities such as:

* optical depth / transmittance;
* cached multiple-scattered radiance;
* direct/single-scattered contribution;
* depth or opacity moment information;
* any additional terms required for view-dependent reconstruction.

For example:

```text
BeamTile
{
    screen bounds
    hierarchy level

    transmittance coefficients
    multiple-scattering coefficients
    single-scattering coefficients
    depth/moment coefficients

    residual information
    error bound
}
```

The exact coefficient basis may evolve.

The architectural requirement does not:

> **BeamTiles are produced from the projected transport representation, not from a set of fully ray-marched camera samples.**

---

# What HSTR Is — and Is Not

The repository contains HSTR transport infrastructure.

HSTR may remain extremely valuable as the **world-space transport representation/backend**.

It contains machinery such as:

* hierarchical transport nodes;
* transmittance pages;
* solved lighting information;
* residual representation;
* adaptive cuts;
* error information;
* world-space cache data.

That does **not** mean the final camera renderer should run an HSTR density march for every Beam sample.

The distinction is:

```text
HSTR
=
world-space transport representation


Beam
=
final camera-space rendering architecture
```

The intended relationship is:

```text
HSTR transport representation
          ↓
projected transport cut
          ↓
BeamTile coefficient construction
          ↓
screen reconstruction
```

NOT:

```text
Beam query
    ↓
march through HSTR/cloud density
    ↓
BeamResult
```

If normal Beam rendering still spends most of its time inside a function such as:

```cpp
marchBeam(...)
```

then the architecture has not been implemented yet.

---

# Existing Projected-Cut Infrastructure

The repository already contains important pieces of the intended architecture.

Examples include:

```text
projectCutNodes
binCutNodes
sortTileCuts
HSTRCutNode
hstrNodeProjection
hstrNodeDepth
hstrNodeOrder
hstrTileNodes
transmittance pages
residual pages
camera lighting
```

This infrastructure projects the solved transport hierarchy into screen-space regions and builds ordered per-tile node lists.

That concept is much closer to the final design than running independent density marches at Beam sample positions.

Existing functionality such as:

```cpp
integratePage(...)
```

is important because it demonstrates the intended principle:

> Accepted hierarchy nodes should be evaluated from their compact transport representation rather than reconstructed by resampling the original volume.

However, even a design that runs a full `traverseCut()` independently for many Beam basis rays should be regarded as an **intermediate step**, not necessarily the endpoint.

The final design should attempt to amortize work across the entire BeamTile.

---

# Final Work Unit

The desired work item is approximately:

```text
BeamTile × ProjectedTransportNode
```

not:

```text
BeamSample × RayStep
```

A projected node should contribute directly to a tile's coefficients.

Conceptually:

```text
for each visible accepted transport node
    project node to screen

    for each overlapping BeamTile
        determine node/tile transfer
        accumulate contribution into BeamTile coefficients
```

This enables neighbouring pixels and basis locations to share transport work.

The expensive hierarchy discovery and transport representation are therefore amortized spatially.

---

# Residual Detail

The compact transport representation will not perfectly describe every part of every cloud.

That is expected.

High-frequency or insufficiently represented regions must be handled as **exceptions**, not as the normal rendering path.

The intended structure is:

```text
projected node
     ↓
representation satisfies error bound?
     │
     ├── yes
     │    ↓
     │ direct BeamTile coefficient contribution
     │
     └── no
          ↓
     residual descriptor
          ↓
     compact residual work queue
```

Residual processing may perform expensive fine-density evaluation when genuinely necessary.

But:

> **Fine marching is a residual path, not the renderer.**

If 20%, 40%, or 80% of the image falls into the residual marcher, the answer is not to optimize that marcher forever.

The representation or error model needs improvement.

---

# Required Complexity Change

A conventional renderer tends toward cost resembling:

```text
screen samples
×
steps per ray
×
density lookup cost
×
lighting lookup cost
```

The final renderer should instead approach:

```text
projected cut-node / BeamTile overlaps
+
sparse residual work
+
cheap full-resolution reconstruction
```

This complexity change is the entire point of the project.

An optimization that merely makes:

```text
N expensive rays
```

become:

```text
0.7N expensive rays
```

may be useful diagnostically, but it is not the central research result.

---

# Sparse Beam Hierarchy

The adaptive screen hierarchy remains useful.

For example:

```text
32×32
  ↓
16×16
  ↓
8×8
  ↓
4×4
```

But subdivision must represent **adaptive BeamTile coefficient resolution**, not simply choose how densely to call `marchBeam()`.

A region should remain coarse when its projected transport can be represented within the error budget.

A region should subdivide when:

* projected transport complexity is too high;
* coefficient approximation error is too large;
* silhouettes require additional resolution;
* residual frequency requires additional resolution;
* depth/transmittance variation requires it.

Therefore:

```text
adaptive Beam hierarchy
```

is part of the final architecture.

```text
adaptive hierarchy of exact ray-march locations
```

is not.

---

# No Dense Lattice as the Work Definition

A regular screen lattice must not define which expensive work exists.

The old idea:

```text
screen lattice
    ↓
corners and centres
    ↓
queries
```

must not be the foundation of the final renderer.

A dense texture may temporarily exist for:

* debugging;
* compatibility comparison;
* visualization;
* validation.

But it must not determine the renderer's workload.

The authoritative structures should eventually be sparse structures such as:

```text
BeamTile[]
ProjectedNodeWork[]
ResidualWork[]
BeamCoefficient[]
```

or equivalent representations.

---

# `marchBeam()` Is Not the Final Renderer

This rule is deliberately explicit because it has already caused architectural confusion.

`marchBeam()` performs conventional camera-ray volume integration.

At the time of writing it ultimately reaches code equivalent to:

```text
marchSegmentMoment
    ↓
majorant lookup
    ↓
step
    ↓
cameraExtinctionAtVoxel
    ↓
brick/page lookup
    ↓
density
    ↓
lighting
    ↓
sun
    ↓
repeat
```

This is exactly the cost the radical architecture is supposed to avoid.

Therefore:

## `marchBeam()` may be used for

* reference comparisons;
* debugging;
* validation;
* residual/fallback work;
* temporary migration A/Bs.

## `marchBeam()` must NOT be

* the normal Beam evaluator;
* the source of every BeamTile coefficient;
* the implementation behind ordinary adaptive Beam queries;
* optimized indefinitely as if it were the final architecture.

If an optimization proposal begins with:

> “make `marchBeam()` faster…”

ask first whether the proposed work belongs in the final residual path.

If not, stop.

---

# Exact Basis Rays Are Also Not the End Goal

A transitional architecture may evaluate:

```text
corner
corner
corner
corner
centre
```

using the projected HSTR cut.

That is already better than marching raw density because accepted nodes can use compact page integration.

But the final design should go further.

Instead of:

```text
five independent traversals of the same projected node list
```

prefer:

```text
one tile/node interaction
     ↓
derive contribution to the tile's basis coefficients
```

This is the deeper source of cross-ray reuse.

Do not stop merely because `marchBeam()` has been replaced by `traverseCut()`.

---

# Lighting

Lighting should follow the same philosophy.

Expensive lighting work should be solved or cached in world space whenever possible.

The ordinary Beam construction path should consume compact lighting/transport information.

Rare cases requiring expensive live lighting should go into a separate exceptional-work path.

Do not place rare fallback logic inside the normal high-frequency kernel if doing so increases:

* register pressure;
* divergence;
* occupancy cost;
* code size;
* instruction-cache pressure.

---

# Empty Space

Empty-space acceleration remains useful, particularly for residual work.

However, do not confuse:

```text
faster ray marching
```

with:

```text
eliminating ray marching
```

Distance-to-density structures, majorants, occupancy hierarchies and similar accelerators are valuable for the residual path.

They do not replace the central Beam architecture.

---

# Temporal Reuse

Temporal reuse is a later multiplier, not the foundation of correctness.

The final system may reuse:

* BeamTile coefficients;
* projected cuts;
* residual classifications;
* prior tile refinement decisions.

Temporal data may help scheduling or reconstruction.

It must not be required to hide an intrinsically expensive steady-state architecture.

First make a fresh frame fast.

Then exploit time.

---

# Quality Authority

The quality authority is the saved path-traced reference.

Not:

* agreement with the old marcher;
* agreement with `marchBeam()`;
* agreement with an old dense lattice;
* exact-match percentage against a legacy implementation.

Legacy exact-march comparisons are useful during development, but they must not force the new architecture to reproduce errors or unnecessary detail from the old renderer.

Report, where available:

* mean error;
* share above the configured error threshold;
* p99;
* p99.9;
* maximum error;
* spatial/block metrics.

The project should preserve the required perceptual/path-reference quality, not mathematical loyalty to legacy code.

---

# Performance Target

Primary target:

```text
3840 × 2160
< 2 ms GPU
```

The renderer must be evaluated under:

* parked/static camera;
* slow camera motion;
* fast camera motion;
* residency churn;
* close cloud views;
* distant cloud views;
* representative lighting conditions.

A static-camera cache hit is not sufficient.

Neither is a single easy camera view.

---

# Performance Budget Philosophy

The desired final cost distribution looks more like:

```text
project / classify transport       small
BeamTile/node accumulation         dominant but cheap
residual work                      sparse
reconstruction                     small
residency maintenance              very small
```

It must NOT look like:

```text
exact camera marching              70–90%
everything else                    10–30%
```

If profiling shows exact camera integration dominating the final renderer, the architecture is wrong or incomplete.

---

# Architectural Invariants

These rules are non-negotiable unless this README is explicitly changed.

### 1. Camera rays are not the normal unit of work.

### 2. `marchBeam()` is not the final Beam evaluator.

### 3. A dense lattice must not define expensive work.

### 4. Accepted world-space transport must be reused directly.

### 5. Expensive fine-density evaluation belongs in a sparse residual path.

### 6. BeamTiles must eventually consume projected transport directly.

### 7. Quality is judged against the path-traced reference.

### 8. Optimizations should reduce algorithmic work, not merely shuffle or micro-optimize millions of equivalent operations.

### 9. 4K output does not imply 4K volumetric integration work.

### 10. The final architecture must materially change scaling relative to conventional ray marching.

---

# AI / Coding-Agent Rules

Before implementing any substantial optimization, answer:

```text
1. Which FINAL architecture component does this advance?

2. Does it reduce dependence on camera-ray marching?

3. Will this code still be useful after marchBeam() stops being the normal renderer?

4. Is this final infrastructure, or explicitly temporary compatibility code?

5. Does it move complexity from
   rays × steps
   toward
   projected transport × tiles?
```

If the answer to #3 is **no**, do not spend significant development time on it without explicit instruction.

---

## Automatic Red Flags

An AI agent should stop and reconsider if its proposed design says things like:

> “We can make each Beam ray cheaper…”

> “Let's optimize `marchBeam()` first…”

> “Emit sparse Beam queries, then run the existing exact march…”

> “Use a coarser lattice to reduce the number of full marches…”

> “Tune the verifier until fewer rays fall back…”

> “Improve the density lookup for every Beam query…”

These can be diagnostic experiments.

They are not the final architecture.

---

# Transitional Code

Some transitional infrastructure is acceptable.

Examples:

```text
legacy-vs-new A/B switch
compatibility lattice output
exact marcher reference path
debug visualizations
temporary basis-ray evaluation
```

Such code should be clearly marked:

```text
TEMPORARY_COMPAT
```

or equivalently documented.

Do not allow temporary compatibility code to become the architecture by inertia.

---

# Current Direction

The implementation path should move toward:

```text
1. Keep world-space transport/residency infrastructure.

2. Produce a camera-visible adaptive transport cut.

3. Project accepted nodes into screen space.

4. Bin/sort them into adaptive BeamTile regions.

5. Replace normal Beam ray marches with direct page/transport evaluation.

6. Convert repeated per-basis cut traversal into tile × node coefficient accumulation.

7. Generate residual work only where the compact representation cannot satisfy the error bound.

8. Reconstruct the final 4K image from BeamTile coefficients.

9. Add temporal reuse only after fresh-frame performance is strong.

10. Optimize residual marching independently because it is no longer the common path.
```

---

# The Architecture Test

At any point, profile the renderer.

If the result looks like:

```text
10 ms  camera / Beam marching
0.2 ms verification
0.5 ms classification
0.8 ms reconstruction
```

the renderer has **not reached the intended architecture**.

The intended result should eventually look structurally more like:

```text
projected transport work
Beam coefficient generation
sparse exceptional residuals
cheap reconstruction
```

with no large normal-path camera marching stage.

---

# One-Sentence Definition

If there is any doubt about the project direction, use this:

> **The final renderer projects a solved hierarchical cloud-transport representation into adaptive screen-space BeamTiles and directly constructs their radiometric coefficients; conventional density ray marching exists only for sparse unresolved residuals, not as the normal way Beam samples are evaluated.**

That is the architecture.
