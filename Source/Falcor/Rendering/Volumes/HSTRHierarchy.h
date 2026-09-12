/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 ***************************************************************************/
#pragma once

#include "HSTROperator.h"

#include <filesystem>

namespace Falcor::hstr
{
struct FALCOR_API HierarchyNode
{
    static constexpr uint32_t kInvalid = ~0u;

    uint32_t left = kInvalid;
    uint32_t right = kInvalid;
    uint32_t parent = kInvalid;
    uint32_t splitAxis = 0;
    DenseMatrix transport;
    DenseMatrix leftInput;
    DenseMatrix rightInput;
    DenseMatrix interfaceSystem;
    DenseMatrix residual;
    float residualNorm = 0.f;
    float conservationError = 0.f;

    bool isLeaf() const { return left == kInvalid; }
};

enum class UpdateStrategy
{
    Woodbury,
    WoodburyKrylov,
    SubtreeRepair,
    FullRefactorization,
};

struct FALCOR_API RankedCorrection
{
    uint32_t node = 0;
    uint16_t row = 0;
    uint16_t col = 0;
    float value = 0.f;
    float goalError = 0.f;
    float score = 0.f;
};

/** Persistent six-face Schur hierarchy with one or more retained modes per face.
 *
 * Parent traces are conservative: side-face incident values are prolonged to
 * both children and outgoing values are restricted with equal weights. Shared
 * child interfaces are eliminated exactly at the retained trace resolution.
 * A 6x6 leaf operator selects the original P0 path. Higher orders retain a
 * uniform spatial sample grid on each directional face.
 */
class FALCOR_API Hierarchy
{
public:
    static Hierarchy compile(uint3 leafDims, const std::vector<DenseMatrix>& leafTransport, uint32_t faceSpatialOrder = 1);

    std::vector<float3> solveFaces(const DenseMatrix& rootIncident) const;
    std::vector<float3> solve(const DenseMatrix& rootIncident) const;
    std::vector<DenseMatrix> getLeafTransferMatrices() const;
    std::vector<DenseMatrix> getLeafSourceToRootMatrices() const;
    std::vector<DenseMatrix> getLeafAdjointGoalMatrices(const DenseMatrix& rootGoals) const;
    std::vector<DenseMatrix> getLeafTransportMatrices() const;
    DenseMatrix solveAdjoint(const DenseMatrix& rootGoal) const;
    std::vector<RankedCorrection> rankResidualAtoms(const DenseMatrix& rootIncident, const DenseMatrix& rootGoal) const;
    std::vector<float> getLeafResidualBounds() const;
    uint32_t updateLeaf(uint32_t leafIndex, const DenseMatrix& transport, UpdateStrategy strategy = UpdateStrategy::SubtreeRepair);

    void save(const std::filesystem::path& path) const;
    static Hierarchy load(const std::filesystem::path& path);

    uint3 getLeafDims() const { return mLeafDims; }
    uint32_t getRoot() const { return mRoot; }
    uint32_t getFaceSpatialOrder() const { return mFaceSpatialOrder; }
    const std::vector<HierarchyNode>& getNodes() const { return mNodes; }

private:
    uint3 mLeafDims = uint3(0);
    uint32_t mRoot = HierarchyNode::kInvalid;
    uint32_t mFaceSpatialOrder = 1;
    std::vector<uint32_t> mLeafNodes;
    std::vector<HierarchyNode> mNodes;
};

/** Energy-conserving P0 leaf response with a separated forward channel. */
FALCOR_API DenseMatrix makeLeafTransport(float3 opticalDepth, float albedo, float anisotropy, float forwardFraction);

/** Exact boundary operator of a cubic grid of six-direction cell operators.
 * The result stores cellsPerAxis^2 spatial samples on each of six faces.
 */
FALCOR_API DenseMatrix makeGridBoundaryTransport(uint32_t cellsPerAxis, const std::vector<DenseMatrix>& cellTransport);
} // namespace Falcor::hstr
