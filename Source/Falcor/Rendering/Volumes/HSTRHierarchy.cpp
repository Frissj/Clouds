/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 ***************************************************************************/
#include "HSTRHierarchy.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

namespace Falcor::hstr
{
namespace
{
constexpr uint32_t kFaceCount = 6;
constexpr uint32_t kFileVersion = 3;
constexpr uint32_t kMagic = 0x52545348; // HSTR

void require(bool condition, const char* message)
{
    if (!condition)
        FALCOR_THROW(message);
}

DenseMatrix add(const DenseMatrix& a, const DenseMatrix& b)
{
    require(a.rows() == b.rows() && a.cols() == b.cols(), "Dense matrix add dimension mismatch.");
    DenseMatrix result(a.rows(), a.cols());
    for (size_t row = 0; row < a.rows(); ++row)
        for (size_t col = 0; col < a.cols(); ++col)
            result(row, col) = a(row, col) + b(row, col);
    return result;
}

float frobeniusNorm(const DenseMatrix& matrix)
{
    float sum = 0.f;
    for (float value : matrix.data())
        sum += value * value;
    return std::sqrt(sum);
}

float conservationError(const DenseMatrix& transport)
{
    float error = 0.f;
    for (size_t col = 0; col < transport.cols(); ++col)
    {
        float sum = 0.f;
        for (size_t row = 0; row < transport.rows(); ++row)
            sum += transport(row, col);
        error = std::max(error, std::max(0.f, sum - 1.f));
    }
    return error;
}

HierarchyNode compose(const HierarchyNode& a, const HierarchyNode& b, uint32_t axis)
{
    const uint32_t negativeFace = 2 * axis;
    const uint32_t positiveFace = negativeFace + 1;
    DenseMatrix prolongA(kFaceCount, kFaceCount);
    DenseMatrix prolongB(kFaceCount, kFaceCount);
    DenseMatrix restrictA(kFaceCount, kFaceCount);
    DenseMatrix restrictB(kFaceCount, kFaceCount);

    for (uint32_t face = 0; face < kFaceCount; ++face)
    {
        if (face == negativeFace)
        {
            prolongA(face, face) = 1.f;
            restrictA(face, face) = 1.f;
        }
        else if (face == positiveFace)
        {
            prolongB(face, face) = 1.f;
            restrictB(face, face) = 1.f;
        }
        else
        {
            prolongA(face, face) = 1.f;
            prolongB(face, face) = 1.f;
            restrictA(face, face) = 0.5f;
            restrictB(face, face) = 0.5f;
        }
    }

    DenseMatrix interfaceSystem(2, 2, {1.f, -b.transport(negativeFace, negativeFace), -a.transport(positiveFace, positiveFace), 1.f});
    DenseMatrix interfaceRhs(2, kFaceCount);
    const DenseMatrix bExternal = multiply(b.transport, prolongB);
    const DenseMatrix aExternal = multiply(a.transport, prolongA);
    for (uint32_t face = 0; face < kFaceCount; ++face)
    {
        interfaceRhs(0, face) = bExternal(negativeFace, face);
        interfaceRhs(1, face) = aExternal(positiveFace, face);
    }
    const DenseMatrix interfaceInput = solve(interfaceSystem, interfaceRhs);

    DenseMatrix inputA = prolongA;
    DenseMatrix inputB = prolongB;
    for (uint32_t face = 0; face < kFaceCount; ++face)
    {
        inputA(positiveFace, face) = interfaceInput(0, face);
        inputB(negativeFace, face) = interfaceInput(1, face);
    }

    const DenseMatrix exact = add(multiply(restrictA, multiply(a.transport, inputA)), multiply(restrictB, multiply(b.transport, inputB)));
    DenseMatrix baseline(kFaceCount, kFaceCount);
    for (uint32_t row = 0; row < kFaceCount; ++row)
        for (uint32_t col = 0; col < kFaceCount; ++col)
            baseline(row, col) = 0.5f * (a.transport(row, col) + b.transport(row, col));

    HierarchyNode result;
    result.splitAxis = axis;
    result.transport = exact;
    result.leftInput = inputA;
    result.rightInput = inputB;
    result.residual = subtract(exact, baseline);
    result.residualNorm = frobeniusNorm(result.residual);
    result.conservationError = conservationError(exact);
    return result;
}

size_t linearIndex(uint3 p, uint3 dims)
{
    return size_t(p.x) + size_t(dims.x) * (size_t(p.y) + size_t(dims.y) * size_t(p.z));
}

uint32_t chooseSplitAxis(uint3 dims, const std::vector<uint32_t>& active, const std::vector<HierarchyNode>& nodes)
{
    float bestScore = std::numeric_limits<float>::infinity();
    uint32_t bestAxis = 0;
    for (uint32_t axis = 0; axis < 3; ++axis)
    {
        if ((&dims.x)[axis] <= 1)
            continue;

        const uint32_t negativeFace = 2 * axis;
        const uint32_t positiveFace = negativeFace + 1;
        float coupling = 0.f;
        uint32_t pairs = 0;
        for (uint32_t z = 0; z < dims.z; ++z)
            for (uint32_t y = 0; y < dims.y; ++y)
                for (uint32_t x = 0; x < dims.x; ++x)
                {
                    uint3 p(x, y, z);
                    if (((&p.x)[axis] & 1u) != 0 || (&p.x)[axis] + 1 >= (&dims.x)[axis])
                        continue;
                    uint3 q = p;
                    ++(&q.x)[axis];
                    const auto& a = nodes[active[linearIndex(p, dims)]].transport;
                    const auto& b = nodes[active[linearIndex(q, dims)]].transport;
                    coupling += std::abs(a(positiveFace, positiveFace) * b(negativeFace, negativeFace));
                    ++pairs;
                }
        // Transport-aware nested dissection: prefer weak separators, while a
        // small aspect penalty prevents pathological needle-shaped levels.
        const float score = coupling / std::max(1u, pairs) + 0.01f * float((&dims.x)[axis]);
        if (score < bestScore)
        {
            bestScore = score;
            bestAxis = axis;
        }
    }
    return bestAxis;
}

void writeMatrix(std::ofstream& stream, const DenseMatrix& matrix)
{
    const uint32_t rows = uint32_t(matrix.rows());
    const uint32_t cols = uint32_t(matrix.cols());
    stream.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    stream.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    stream.write(reinterpret_cast<const char*>(matrix.data().data()), matrix.data().size() * sizeof(float));
}

DenseMatrix readMatrix(std::ifstream& stream)
{
    uint32_t rows = 0;
    uint32_t cols = 0;
    stream.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    stream.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    DenseMatrix result(rows, cols);
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t col = 0; col < cols; ++col)
            stream.read(reinterpret_cast<char*>(&result(row, col)), sizeof(float));
    return result;
}
} // namespace

Hierarchy Hierarchy::compile(uint3 leafDims, const std::vector<DenseMatrix>& leafTransport)
{
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;
    require(leafDims.x > 0 && leafDims.y > 0 && leafDims.z > 0, "HST-R leaf dimensions must be non-zero.");
    require(leafTransport.size() == leafCount, "HST-R leaf transport count does not match dimensions.");
    require(isPowerOf2(leafDims.x) && isPowerOf2(leafDims.y) && isPowerOf2(leafDims.z), "HST-R leaf dimensions must be powers of two.");

    Hierarchy hierarchy;
    hierarchy.mLeafDims = leafDims;
    hierarchy.mNodes.reserve(2 * leafCount);
    hierarchy.mLeafNodes.reserve(leafCount);
    std::vector<uint32_t> active;
    active.reserve(leafCount);
    for (const DenseMatrix& transport : leafTransport)
    {
        require(transport.rows() == kFaceCount && transport.cols() == kFaceCount, "HST-R leaf operators must be 6x6.");
        HierarchyNode node;
        node.transport = transport;
        node.conservationError = conservationError(transport);
        const uint32_t id = uint32_t(hierarchy.mNodes.size());
        hierarchy.mNodes.push_back(std::move(node));
        hierarchy.mLeafNodes.push_back(id);
        active.push_back(id);
    }

    uint3 dims = leafDims;
    while (dims.x * dims.y * dims.z > 1)
    {
        const uint32_t axis = chooseSplitAxis(dims, active, hierarchy.mNodes);
        uint3 nextDims = dims;
        (&nextDims.x)[axis] /= 2;
        std::vector<uint32_t> next(size_t(nextDims.x) * nextDims.y * nextDims.z);
        for (uint32_t z = 0; z < nextDims.z; ++z)
            for (uint32_t y = 0; y < nextDims.y; ++y)
                for (uint32_t x = 0; x < nextDims.x; ++x)
                {
                    uint3 p(x, y, z);
                    uint3 aPos = p;
                    (&aPos.x)[axis] *= 2;
                    uint3 bPos = aPos;
                    ++(&bPos.x)[axis];
                    const uint32_t left = active[linearIndex(aPos, dims)];
                    const uint32_t right = active[linearIndex(bPos, dims)];
                    HierarchyNode parent = compose(hierarchy.mNodes[left], hierarchy.mNodes[right], axis);
                    parent.left = left;
                    parent.right = right;
                    const uint32_t parentID = uint32_t(hierarchy.mNodes.size());
                    hierarchy.mNodes.push_back(std::move(parent));
                    hierarchy.mNodes[left].parent = parentID;
                    hierarchy.mNodes[right].parent = parentID;
                    next[linearIndex(p, nextDims)] = parentID;
                }
        active = std::move(next);
        dims = nextDims;
    }
    hierarchy.mRoot = active.front();
    return hierarchy;
}

std::vector<float3> Hierarchy::solveFaces(const DenseMatrix& rootIncident) const
{
    require(rootIncident.rows() == kFaceCount && rootIncident.cols() == 3, "HST-R root incident field must be 6x3 RGB.");
    require(mRoot != HierarchyNode::kInvalid, "HST-R hierarchy is empty.");
    std::vector<DenseMatrix> incident(mNodes.size());
    incident[mRoot] = rootIncident;
    std::vector<uint32_t> stack{mRoot};
    while (!stack.empty())
    {
        const uint32_t id = stack.back();
        stack.pop_back();
        const HierarchyNode& node = mNodes[id];
        if (node.isLeaf())
            continue;
        incident[node.left] = multiply(node.leftInput, incident[id]);
        incident[node.right] = multiply(node.rightInput, incident[id]);
        stack.push_back(node.left);
        stack.push_back(node.right);
    }

    std::vector<float3> result(mLeafNodes.size() * kFaceCount);
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
    {
        const uint32_t id = mLeafNodes[leaf];
        const DenseMatrix outgoing = multiply(mNodes[id].transport, incident[id]);
        for (uint32_t face = 0; face < kFaceCount; ++face)
            for (uint32_t channel = 0; channel < 3; ++channel)
                result[leaf * kFaceCount + face][channel] = outgoing(face, channel);
    }
    return result;
}

std::vector<float3> Hierarchy::solve(const DenseMatrix& rootIncident) const
{
    const std::vector<float3> faces = solveFaces(rootIncident);
    std::vector<float3> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        for (uint32_t face = 0; face < kFaceCount; ++face)
            result[leaf] += faces[leaf * kFaceCount + face] / float(kFaceCount);
    return result;
}

std::vector<DenseMatrix> Hierarchy::getLeafTransferMatrices() const
{
    require(mRoot != HierarchyNode::kInvalid, "HST-R hierarchy is empty.");
    std::vector<DenseMatrix> transfer(mNodes.size());
    transfer[mRoot] = DenseMatrix::identity(kFaceCount);
    std::vector<uint32_t> stack{mRoot};
    while (!stack.empty())
    {
        const uint32_t id = stack.back();
        stack.pop_back();
        const HierarchyNode& node = mNodes[id];
        if (node.isLeaf())
            continue;
        transfer[node.left] = multiply(node.leftInput, transfer[id]);
        transfer[node.right] = multiply(node.rightInput, transfer[id]);
        stack.push_back(node.left);
        stack.push_back(node.right);
    }
    std::vector<DenseMatrix> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        result[leaf] = std::move(transfer[mLeafNodes[leaf]]);
    return result;
}

std::vector<DenseMatrix> Hierarchy::getLeafTransportMatrices() const
{
    std::vector<DenseMatrix> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        result[leaf] = mNodes[mLeafNodes[leaf]].transport;
    return result;
}

DenseMatrix Hierarchy::solveAdjoint(const DenseMatrix& rootGoal) const
{
    require(rootGoal.rows() == kFaceCount, "HST-R adjoint goal must have six face rows.");
    return multiply(transpose(mNodes[mRoot].transport), rootGoal);
}

std::vector<RankedCorrection> Hierarchy::rankResidualAtoms(const DenseMatrix& rootIncident, const DenseMatrix& rootGoal) const
{
    require(rootIncident.rows() == kFaceCount, "HST-R root incident field must have six face rows.");
    require(rootGoal.rows() == kFaceCount && rootGoal.cols() == rootIncident.cols(), "HST-R root goal dimensions do not match.");

    std::vector<RankedCorrection> atoms;
    struct Work
    {
        uint32_t node;
        DenseMatrix incident;
        DenseMatrix goal;
    };
    std::vector<Work> stack;
    stack.push_back({mRoot, rootIncident, rootGoal});
    while (!stack.empty())
    {
        Work work = std::move(stack.back());
        stack.pop_back();
        const HierarchyNode& node = mNodes[work.node];
        if (node.isLeaf())
            continue;

        for (uint16_t row = 0; row < node.residual.rows(); ++row)
            for (uint16_t col = 0; col < node.residual.cols(); ++col)
            {
                const float value = node.residual(row, col);
                float goalError = 0.f;
                for (size_t channel = 0; channel < work.incident.cols(); ++channel)
                    goalError += std::abs(work.goal(row, channel) * value * work.incident(col, channel));
                if (goalError > 0.f)
                    atoms.push_back({work.node, row, col, value, goalError, goalError / sizeof(RankedCorrection)});
            }

        DenseMatrix leftRestriction(kFaceCount, kFaceCount);
        DenseMatrix rightRestriction(kFaceCount, kFaceCount);
        const uint32_t negativeFace = 2 * node.splitAxis;
        const uint32_t positiveFace = negativeFace + 1;
        for (uint32_t face = 0; face < kFaceCount; ++face)
        {
            if (face == negativeFace)
                leftRestriction(face, face) = 1.f;
            else if (face == positiveFace)
                rightRestriction(face, face) = 1.f;
            else
            {
                leftRestriction(face, face) = 0.5f;
                rightRestriction(face, face) = 0.5f;
            }
        }
        stack.push_back({
            node.left,
            multiply(node.leftInput, work.incident),
            multiply(transpose(leftRestriction), work.goal),
        });
        stack.push_back({
            node.right,
            multiply(node.rightInput, work.incident),
            multiply(transpose(rightRestriction), work.goal),
        });
    }
    std::stable_sort(atoms.begin(), atoms.end(), [](const RankedCorrection& a, const RankedCorrection& b) { return a.score > b.score; });
    return atoms;
}

std::vector<float> Hierarchy::getLeafResidualBounds() const
{
    std::vector<float> nodeBounds(mNodes.size(), 0.f);
    std::vector<uint32_t> stack{mRoot};
    while (!stack.empty())
    {
        const uint32_t id = stack.back();
        stack.pop_back();
        const HierarchyNode& node = mNodes[id];
        if (node.isLeaf())
            continue;
        nodeBounds[node.left] = nodeBounds[id] + node.residualNorm;
        nodeBounds[node.right] = nodeBounds[id] + node.residualNorm;
        stack.push_back(node.left);
        stack.push_back(node.right);
    }
    std::vector<float> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        result[leaf] = nodeBounds[mLeafNodes[leaf]];
    return result;
}

uint32_t Hierarchy::updateLeaf(uint32_t leafIndex, const DenseMatrix& transport)
{
    require(leafIndex < mLeafNodes.size(), "HST-R leaf update index is out of range.");
    require(transport.rows() == kFaceCount && transport.cols() == kFaceCount, "HST-R leaf update must be 6x6.");
    uint32_t nodeID = mLeafNodes[leafIndex];
    mNodes[nodeID].transport = transport;
    uint32_t repaired = 0;
    while (mNodes[nodeID].parent != HierarchyNode::kInvalid)
    {
        const uint32_t parentID = mNodes[nodeID].parent;
        const HierarchyNode oldParent = mNodes[parentID];
        HierarchyNode updated = compose(mNodes[oldParent.left], mNodes[oldParent.right], oldParent.splitAxis);
        updated.left = oldParent.left;
        updated.right = oldParent.right;
        updated.parent = oldParent.parent;
        mNodes[parentID] = std::move(updated);
        nodeID = parentID;
        ++repaired;
    }
    return repaired;
}

void Hierarchy::save(const std::filesystem::path& path) const
{
    std::ofstream stream(path, std::ios::binary);
    require(bool(stream), "Failed to create HST-R hierarchy file.");
    stream.write(reinterpret_cast<const char*>(&kMagic), sizeof(kMagic));
    stream.write(reinterpret_cast<const char*>(&kFileVersion), sizeof(kFileVersion));
    stream.write(reinterpret_cast<const char*>(&mLeafDims), sizeof(mLeafDims));
    stream.write(reinterpret_cast<const char*>(&mRoot), sizeof(mRoot));
    const uint32_t leafCount = uint32_t(mLeafNodes.size());
    const uint32_t nodeCount = uint32_t(mNodes.size());
    stream.write(reinterpret_cast<const char*>(&leafCount), sizeof(leafCount));
    stream.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
    stream.write(reinterpret_cast<const char*>(mLeafNodes.data()), mLeafNodes.size() * sizeof(uint32_t));
    for (const HierarchyNode& node : mNodes)
    {
        stream.write(reinterpret_cast<const char*>(&node.left), sizeof(node.left));
        stream.write(reinterpret_cast<const char*>(&node.right), sizeof(node.right));
        stream.write(reinterpret_cast<const char*>(&node.parent), sizeof(node.parent));
        stream.write(reinterpret_cast<const char*>(&node.splitAxis), sizeof(node.splitAxis));
        stream.write(reinterpret_cast<const char*>(&node.residualNorm), sizeof(node.residualNorm));
        stream.write(reinterpret_cast<const char*>(&node.conservationError), sizeof(node.conservationError));
        writeMatrix(stream, node.transport);
        writeMatrix(stream, node.leftInput);
        writeMatrix(stream, node.rightInput);
        writeMatrix(stream, node.residual);
    }
}

Hierarchy Hierarchy::load(const std::filesystem::path& path)
{
    std::ifstream stream(path, std::ios::binary);
    require(bool(stream), "Failed to open HST-R hierarchy file.");
    uint32_t magic = 0;
    uint32_t version = 0;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    stream.read(reinterpret_cast<char*>(&version), sizeof(version));
    require(magic == kMagic && version == kFileVersion, "Unsupported HST-R hierarchy file.");

    Hierarchy hierarchy;
    stream.read(reinterpret_cast<char*>(&hierarchy.mLeafDims), sizeof(hierarchy.mLeafDims));
    stream.read(reinterpret_cast<char*>(&hierarchy.mRoot), sizeof(hierarchy.mRoot));
    uint32_t leafCount = 0;
    uint32_t nodeCount = 0;
    stream.read(reinterpret_cast<char*>(&leafCount), sizeof(leafCount));
    stream.read(reinterpret_cast<char*>(&nodeCount), sizeof(nodeCount));
    hierarchy.mLeafNodes.resize(leafCount);
    hierarchy.mNodes.resize(nodeCount);
    stream.read(reinterpret_cast<char*>(hierarchy.mLeafNodes.data()), hierarchy.mLeafNodes.size() * sizeof(uint32_t));
    for (HierarchyNode& node : hierarchy.mNodes)
    {
        stream.read(reinterpret_cast<char*>(&node.left), sizeof(node.left));
        stream.read(reinterpret_cast<char*>(&node.right), sizeof(node.right));
        stream.read(reinterpret_cast<char*>(&node.parent), sizeof(node.parent));
        stream.read(reinterpret_cast<char*>(&node.splitAxis), sizeof(node.splitAxis));
        stream.read(reinterpret_cast<char*>(&node.residualNorm), sizeof(node.residualNorm));
        stream.read(reinterpret_cast<char*>(&node.conservationError), sizeof(node.conservationError));
        node.transport = readMatrix(stream);
        node.leftInput = readMatrix(stream);
        node.rightInput = readMatrix(stream);
        node.residual = readMatrix(stream);
    }
    require(bool(stream), "HST-R hierarchy file is truncated.");
    return hierarchy;
}

DenseMatrix makeLeafTransport(float3 opticalDepth, float albedo, float anisotropy, float forwardFraction)
{
    albedo = std::clamp(albedo, 0.f, 1.f);
    anisotropy = std::clamp(anisotropy, -0.99f, 0.99f);
    forwardFraction = std::clamp(forwardFraction, 0.f, 0.99f);
    const float3 normals[] = {
        float3(-1, 0, 0),
        float3(1, 0, 0),
        float3(0, -1, 0),
        float3(0, 1, 0),
        float3(0, 0, -1),
        float3(0, 0, 1),
    };

    DenseMatrix result(kFaceCount, kFaceCount);
    for (uint32_t input = 0; input < kFaceCount; ++input)
    {
        const uint32_t axis = input / 2;
        const uint32_t opposite = input ^ 1u;
        const float reducedDepth = std::max(0.f, opticalDepth[axis]) * (1.f - albedo * forwardFraction);
        const float ballistic = std::exp(-reducedDepth);
        const float scattered = albedo * (1.f - ballistic);
        // High-order radiance becomes angularly smooth in optically thick
        // regions. Retain the ballistic channel and fade only the residual
        // scattering lobe toward its diffusion limit.
        const float diffusionWeight = std::clamp((reducedDepth - 1.5f) / 3.f, 0.f, 1.f);
        const float residualAnisotropy = anisotropy * (1.f - diffusionWeight);
        float weights[kFaceCount];
        float weightSum = 0.f;
        for (uint32_t output = 0; output < kFaceCount; ++output)
        {
            const float cosine = dot(-normals[input], normals[output]);
            const float denominator = std::max(1e-4f, 1.f + residualAnisotropy * residualAnisotropy - 2.f * residualAnisotropy * cosine);
            weights[output] = (1.f - residualAnisotropy * residualAnisotropy) / (denominator * std::sqrt(denominator));
            weightSum += weights[output];
        }
        for (uint32_t output = 0; output < kFaceCount; ++output)
            result(output, input) = scattered * weights[output] / weightSum;
        result(opposite, input) += ballistic;
    }
    return result;
}
} // namespace Falcor::hstr
