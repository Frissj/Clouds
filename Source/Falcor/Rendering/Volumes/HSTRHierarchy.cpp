/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 ***************************************************************************/
#include "HSTRHierarchy.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <limits>
#include <mutex>
#include <thread>

namespace Falcor::hstr
{
namespace
{
constexpr uint32_t kFaceCount = 6;
constexpr uint32_t kFileVersion = 5;
constexpr uint32_t kMagic = 0x52545348; // HSTR

void require(bool condition, const char* message)
{
    if (!condition)
        FALCOR_THROW(message);
}

thread_local bool tInsideParallelFor = false;

/// Runs f(i) for i in [0, count) on all hardware threads and rethrows the first worker exception.
/// Nested calls (for example compiling many small hierarchies from a parallel loop) run serially.
template<typename F>
void parallelFor(size_t count, F&& f)
{
    if (tInsideParallelFor)
    {
        for (size_t i = 0; i < count; ++i)
            f(i);
        return;
    }
    const size_t threadCount = std::min<size_t>(std::max(1u, std::thread::hardware_concurrency()), count);
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::mutex errorMutex;
    auto worker = [&]()
    {
        const bool wasInside = tInsideParallelFor;
        tInsideParallelFor = true;
        try
        {
            for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1))
                f(i);
        }
        catch (...)
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (!error)
                error = std::current_exception();
            next = count;
        }
        tInsideParallelFor = wasInside;
    };
    std::vector<std::thread> threads;
    for (size_t t = 1; t < threadCount; ++t)
        threads.emplace_back(worker);
    worker();
    for (auto& thread : threads)
        thread.join();
    if (error)
        std::rethrow_exception(error);
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

size_t faceDofs(const DenseMatrix& transport)
{
    require(transport.rows() == transport.cols() && transport.rows() % kFaceCount == 0, "HST-R operators must be square with equal per-face dimensions.");
    return transport.rows() / kFaceCount;
}

size_t faceOffset(uint32_t face, size_t dofsPerFace)
{
    return size_t(face) * dofsPerFace;
}

void makeChildTraceTransfer(
    uint32_t splitAxis,
    uint32_t childSide,
    uint32_t spatialOrder,
    DenseMatrix& prolongation,
    DenseMatrix& restriction
)
{
    const size_t dofsPerFace = size_t(spatialOrder) * spatialOrder;
    const size_t traceDofs = kFaceCount * dofsPerFace;
    prolongation = DenseMatrix(traceDofs, traceDofs);
    restriction = DenseMatrix(traceDofs, traceDofs);
    for (uint32_t face = 0; face < kFaceCount; ++face)
    {
        const uint32_t faceAxis = face / 2;
        if (faceAxis == splitAxis)
        {
            if ((face & 1u) != childSide)
                continue;
            for (size_t mode = 0; mode < dofsPerFace; ++mode)
            {
                const size_t dof = faceOffset(face, dofsPerFace) + mode;
                prolongation(dof, dof) = 1.f;
                restriction(dof, dof) = 1.f;
            }
            continue;
        }

        uint32_t tangents[2];
        uint32_t tangentCount = 0;
        for (uint32_t axis = 0; axis < 3; ++axis)
            if (axis != faceAxis)
                tangents[tangentCount++] = axis;
        const uint32_t splitCoordinate = tangents[0] == splitAxis ? 0u : 1u;
        for (uint32_t v = 0; v < spatialOrder; ++v)
            for (uint32_t u = 0; u < spatialOrder; ++u)
            {
                uint32_t parentU = u;
                uint32_t parentV = v;
                if (splitCoordinate == 0)
                    parentU = (childSide * spatialOrder + u) / 2;
                else
                    parentV = (childSide * spatialOrder + v) / 2;
                const size_t childDof = faceOffset(face, dofsPerFace) + size_t(v) * spatialOrder + u;
                const size_t parentDof = faceOffset(face, dofsPerFace) + size_t(parentV) * spatialOrder + parentU;
                prolongation(childDof, parentDof) = 1.f;
                restriction(parentDof, childDof) = 0.5f;
            }
    }
}

HierarchyNode compose(
    const HierarchyNode& a,
    const HierarchyNode& b,
    uint32_t axis,
    uint32_t spatialOrder,
    const DenseMatrix* pPreviousSystem = nullptr,
    UpdateStrategy strategy = UpdateStrategy::FullRefactorization
)
{
    require(a.transport.rows() == b.transport.rows(), "HST-R child trace dimensions must match.");
    const size_t dofsPerFace = faceDofs(a.transport);
    const size_t traceDofs = kFaceCount * dofsPerFace;
    const uint32_t negativeFace = 2 * axis;
    const uint32_t positiveFace = negativeFace + 1;
    require(dofsPerFace == size_t(spatialOrder) * spatialOrder, "HST-R trace dimensions do not match the face spatial order.");
    DenseMatrix prolongA;
    DenseMatrix prolongB;
    DenseMatrix restrictA;
    DenseMatrix restrictB;
    makeChildTraceTransfer(axis, 0, spatialOrder, prolongA, restrictA);
    makeChildTraceTransfer(axis, 1, spatialOrder, prolongB, restrictB);

    DenseMatrix interfaceSystem(2 * dofsPerFace, 2 * dofsPerFace);
    DenseMatrix interfaceRhs(2 * dofsPerFace, traceDofs);
    const DenseMatrix bExternal = multiply(b.transport, prolongB);
    const DenseMatrix aExternal = multiply(a.transport, prolongA);
    const size_t negativeOffset = faceOffset(negativeFace, dofsPerFace);
    const size_t positiveOffset = faceOffset(positiveFace, dofsPerFace);
    for (size_t row = 0; row < dofsPerFace; ++row)
    {
        interfaceSystem(row, row) = 1.f;
        interfaceSystem(dofsPerFace + row, dofsPerFace + row) = 1.f;
        for (size_t col = 0; col < dofsPerFace; ++col)
        {
            interfaceSystem(row, dofsPerFace + col) = -b.transport(negativeOffset + row, negativeOffset + col);
            interfaceSystem(dofsPerFace + row, col) = -a.transport(positiveOffset + row, positiveOffset + col);
        }
        for (size_t col = 0; col < traceDofs; ++col)
        {
            interfaceRhs(row, col) = bExternal(negativeOffset + row, col);
            interfaceRhs(dofsPerFace + row, col) = aExternal(positiveOffset + row, col);
        }
    }
    DenseMatrix interfaceInput;
    if (pPreviousSystem && strategy == UpdateStrategy::Woodbury)
    {
        const DenseMatrix delta = subtract(interfaceSystem, *pPreviousSystem);
        const LowRankOperator update = compress(delta, 0.f, delta.rows());
        interfaceInput = woodburySolve(*pPreviousSystem, update.left, update.right, interfaceRhs);
    }
    else if (pPreviousSystem && strategy == UpdateStrategy::WoodburyKrylov)
    {
        interfaceInput = DenseMatrix(interfaceRhs.rows(), interfaceRhs.cols());
        for (size_t col = 0; col < interfaceRhs.cols(); ++col)
        {
            DenseMatrix rhs(interfaceRhs.rows(), 1);
            for (size_t row = 0; row < interfaceRhs.rows(); ++row)
                rhs(row, 0) = interfaceRhs(row, col);
            const KrylovResult solution = gmres(
                interfaceSystem.rows(),
                [&interfaceSystem](const DenseMatrix& x) { return multiply(interfaceSystem, x); },
                [pPreviousSystem](const DenseMatrix& x) { return solve(*pPreviousSystem, x); },
                rhs,
                uint32_t(interfaceSystem.rows()),
                1e-5f
            );
            for (size_t row = 0; row < interfaceRhs.rows(); ++row)
                interfaceInput(row, col) = solution.solution(row, 0);
        }
    }
    else
        interfaceInput = solve(interfaceSystem, interfaceRhs);

    DenseMatrix inputA = prolongA;
    DenseMatrix inputB = prolongB;
    for (size_t row = 0; row < dofsPerFace; ++row)
        for (size_t col = 0; col < traceDofs; ++col)
        {
            inputA(positiveOffset + row, col) = interfaceInput(row, col);
            inputB(negativeOffset + row, col) = interfaceInput(dofsPerFace + row, col);
        }

    const DenseMatrix exact = add(multiply(restrictA, multiply(a.transport, inputA)), multiply(restrictB, multiply(b.transport, inputB)));
    DenseMatrix baseline(traceDofs, traceDofs);
    for (size_t row = 0; row < traceDofs; ++row)
        for (size_t col = 0; col < traceDofs; ++col)
            baseline(row, col) = 0.5f * (a.transport(row, col) + b.transport(row, col));

    HierarchyNode result;
    result.splitAxis = axis;
    result.transport = exact;
    result.leftInput = inputA;
    result.rightInput = inputB;
    result.interfaceSystem = interfaceSystem;
    result.residual = subtract(exact, baseline);
    result.residualNorm = frobeniusNorm(result.residual);
    result.conservationError = conservationError(exact);
    return result;
}

size_t linearIndex(uint3 p, uint3 dims)
{
    return size_t(p.x) + size_t(dims.x) * (size_t(p.y) + size_t(dims.y) * size_t(p.z));
}

/// Top-down traversal that visits every internal node after its parent; nodes on one level run concurrently.
template<typename F>
void forEachLevel(const std::vector<HierarchyNode>& nodes, uint32_t root, F&& visit)
{
    std::vector<uint32_t> level{root};
    while (!level.empty())
    {
        parallelFor(
            level.size(),
            [&](size_t i)
            {
                if (!nodes[level[i]].isLeaf())
                    visit(nodes[level[i]], level[i]);
            }
        );
        std::vector<uint32_t> next;
        next.reserve(2 * level.size());
        for (uint32_t id : level)
            if (!nodes[id].isLeaf())
            {
                next.push_back(nodes[id].left);
                next.push_back(nodes[id].right);
            }
        level = std::move(next);
    }
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
                    const size_t aDofs = faceDofs(a);
                    const size_t bDofs = faceDofs(b);
                    require(aDofs == bDofs, "HST-R neighboring trace dimensions must match.");
                    const size_t positiveOffset = faceOffset(positiveFace, aDofs);
                    const size_t negativeOffset = faceOffset(negativeFace, aDofs);
                    float blockCoupling = 0.f;
                    for (size_t row = 0; row < aDofs; ++row)
                        for (size_t k = 0; k < aDofs; ++k)
                            for (size_t col = 0; col < aDofs; ++col)
                            {
                                const float value = a(positiveOffset + row, positiveOffset + k) * b(negativeOffset + k, negativeOffset + col);
                                blockCoupling += value * value;
                            }
                    coupling += std::sqrt(blockCoupling);
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

Hierarchy Hierarchy::compile(uint3 leafDims, const std::vector<DenseMatrix>& leafTransport, uint32_t faceSpatialOrder)
{
    const size_t leafCount = size_t(leafDims.x) * leafDims.y * leafDims.z;
    require(leafDims.x > 0 && leafDims.y > 0 && leafDims.z > 0, "HST-R leaf dimensions must be non-zero.");
    require(leafTransport.size() == leafCount, "HST-R leaf transport count does not match dimensions.");
    require(isPowerOf2(leafDims.x) && isPowerOf2(leafDims.y) && isPowerOf2(leafDims.z), "HST-R leaf dimensions must be powers of two.");
    require(faceSpatialOrder > 0, "HST-R face spatial order must be non-zero.");

    Hierarchy hierarchy;
    hierarchy.mLeafDims = leafDims;
    hierarchy.mFaceSpatialOrder = faceSpatialOrder;
    hierarchy.mNodes.reserve(2 * leafCount);
    hierarchy.mLeafNodes.reserve(leafCount);
    std::vector<uint32_t> active;
    active.reserve(leafCount);
    const size_t traceDofs = leafTransport.front().rows();
    for (const DenseMatrix& transport : leafTransport)
    {
        require(
            traceDofs > 0 && transport.rows() == traceDofs && transport.cols() == traceDofs && traceDofs % kFaceCount == 0,
            "HST-R leaf operators must be non-empty, square, and have matching equal per-face dimensions."
        );
        require(traceDofs / kFaceCount == size_t(faceSpatialOrder) * faceSpatialOrder, "HST-R leaf trace dimensions do not match the face spatial order.");
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
        const size_t nextCount = size_t(nextDims.x) * nextDims.y * nextDims.z;
        std::vector<uint32_t> next(nextCount);
        // Parents on one level are independent; compose them concurrently, then append in the serial order.
        std::vector<HierarchyNode> parents(nextCount);
        parallelFor(
            nextCount,
            [&](size_t i)
            {
                const uint3 p(uint32_t(i % nextDims.x), uint32_t((i / nextDims.x) % nextDims.y), uint32_t(i / (nextDims.x * nextDims.y)));
                uint3 aPos = p;
                (&aPos.x)[axis] *= 2;
                uint3 bPos = aPos;
                ++(&bPos.x)[axis];
                const uint32_t left = active[linearIndex(aPos, dims)];
                const uint32_t right = active[linearIndex(bPos, dims)];
                parents[i] = compose(hierarchy.mNodes[left], hierarchy.mNodes[right], axis, faceSpatialOrder);
                parents[i].left = left;
                parents[i].right = right;
            }
        );
        for (size_t i = 0; i < nextCount; ++i)
        {
            const uint32_t parentID = uint32_t(hierarchy.mNodes.size());
            hierarchy.mNodes[parents[i].left].parent = parentID;
            hierarchy.mNodes[parents[i].right].parent = parentID;
            hierarchy.mNodes.push_back(std::move(parents[i]));
            next[i] = parentID;
        }
        active = std::move(next);
        dims = nextDims;
    }
    hierarchy.mRoot = active.front();
    return hierarchy;
}

std::vector<float3> Hierarchy::solveFaces(const DenseMatrix& rootIncident) const
{
    require(mRoot != HierarchyNode::kInvalid, "HST-R hierarchy is empty.");
    const size_t traceDofs = mNodes[mRoot].transport.rows();
    require(rootIncident.rows() == traceDofs && rootIncident.cols() == 3, "HST-R root incident field dimensions do not match the retained trace.");
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

    std::vector<float3> result(mLeafNodes.size() * traceDofs);
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
    {
        const uint32_t id = mLeafNodes[leaf];
        const DenseMatrix outgoing = multiply(mNodes[id].transport, incident[id]);
        for (size_t dof = 0; dof < traceDofs; ++dof)
            for (uint32_t channel = 0; channel < 3; ++channel)
                result[leaf * traceDofs + dof][channel] = outgoing(dof, channel);
    }
    return result;
}

std::vector<float3> Hierarchy::solve(const DenseMatrix& rootIncident) const
{
    const std::vector<float3> faces = solveFaces(rootIncident);
    const size_t dofsPerFace = faceDofs(mNodes[mRoot].transport);
    const size_t traceDofs = kFaceCount * dofsPerFace;
    std::vector<float3> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        for (uint32_t face = 0; face < kFaceCount; ++face)
            for (size_t mode = 0; mode < dofsPerFace; ++mode)
                result[leaf] += faces[leaf * traceDofs + faceOffset(face, dofsPerFace) + mode] / float(traceDofs);
    return result;
}

std::vector<DenseMatrix> Hierarchy::getLeafTransferMatrices() const
{
    require(mRoot != HierarchyNode::kInvalid, "HST-R hierarchy is empty.");
    std::vector<DenseMatrix> transfer(mNodes.size());
    transfer[mRoot] = DenseMatrix::identity(mNodes[mRoot].transport.rows());
    forEachLevel(
        mNodes,
        mRoot,
        [&](const HierarchyNode& node, uint32_t id)
        {
            transfer[node.left] = multiply(node.leftInput, transfer[id]);
            transfer[node.right] = multiply(node.rightInput, transfer[id]);
        }
    );
    std::vector<DenseMatrix> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        result[leaf] = std::move(transfer[mLeafNodes[leaf]]);
    return result;
}

std::vector<DenseMatrix> Hierarchy::getLeafSourceToRootMatrices() const
{
    std::vector<DenseMatrix> result = getLeafTransferMatrices();
    for (DenseMatrix& transfer : result)
        transfer = transpose(transfer);
    return result;
}

std::vector<DenseMatrix> Hierarchy::getLeafAdjointGoalMatrices(const DenseMatrix& rootGoals) const
{
    require(mRoot != HierarchyNode::kInvalid, "HST-R hierarchy is empty.");
    require(rootGoals.rows() == mNodes[mRoot].transport.rows(), "HST-R root goal dimensions do not match the retained trace.");
    std::vector<DenseMatrix> goals(mNodes.size());
    goals[mRoot] = rootGoals;
    forEachLevel(
        mNodes,
        mRoot,
        [&](const HierarchyNode& node, uint32_t id)
        {
            DenseMatrix leftProlongation;
            DenseMatrix rightProlongation;
            DenseMatrix leftRestriction;
            DenseMatrix rightRestriction;
            makeChildTraceTransfer(node.splitAxis, 0, mFaceSpatialOrder, leftProlongation, leftRestriction);
            makeChildTraceTransfer(node.splitAxis, 1, mFaceSpatialOrder, rightProlongation, rightRestriction);
            goals[node.left] = multiply(transpose(leftRestriction), goals[id]);
            goals[node.right] = multiply(transpose(rightRestriction), goals[id]);
        }
    );

    std::vector<DenseMatrix> result(mLeafNodes.size());
    for (size_t leaf = 0; leaf < mLeafNodes.size(); ++leaf)
        result[leaf] = std::move(goals[mLeafNodes[leaf]]);
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
    require(rootGoal.rows() == mNodes[mRoot].transport.rows(), "HST-R adjoint goal dimensions do not match the retained trace.");
    return multiply(transpose(mNodes[mRoot].transport), rootGoal);
}

std::vector<RankedCorrection> Hierarchy::rankResidualAtoms(const DenseMatrix& rootIncident, const DenseMatrix& rootGoal) const
{
    const size_t traceDofs = mNodes[mRoot].transport.rows();
    require(rootIncident.rows() == traceDofs, "HST-R root incident field dimensions do not match the retained trace.");
    require(rootGoal.rows() == traceDofs && rootGoal.cols() == rootIncident.cols(), "HST-R root goal dimensions do not match.");

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

        DenseMatrix leftProlongation;
        DenseMatrix rightProlongation;
        DenseMatrix leftRestriction;
        DenseMatrix rightRestriction;
        makeChildTraceTransfer(node.splitAxis, 0, mFaceSpatialOrder, leftProlongation, leftRestriction);
        makeChildTraceTransfer(node.splitAxis, 1, mFaceSpatialOrder, rightProlongation, rightRestriction);
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

uint32_t Hierarchy::updateLeaf(uint32_t leafIndex, const DenseMatrix& transport, UpdateStrategy strategy)
{
    require(leafIndex < mLeafNodes.size(), "HST-R leaf update index is out of range.");
    require(transport.rows() == mNodes[mLeafNodes[leafIndex]].transport.rows() && transport.cols() == transport.rows(), "HST-R leaf update dimensions do not match the retained trace.");
    uint32_t nodeID = mLeafNodes[leafIndex];
    mNodes[nodeID].transport = transport;
    uint32_t repaired = 0;
    while (mNodes[nodeID].parent != HierarchyNode::kInvalid)
    {
        const uint32_t parentID = mNodes[nodeID].parent;
        const HierarchyNode oldParent = mNodes[parentID];
        HierarchyNode updated = compose(
            mNodes[oldParent.left],
            mNodes[oldParent.right],
            oldParent.splitAxis,
            mFaceSpatialOrder,
            oldParent.interfaceSystem.rows() > 0 ? &oldParent.interfaceSystem : nullptr,
            strategy
        );
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
    stream.write(reinterpret_cast<const char*>(&mFaceSpatialOrder), sizeof(mFaceSpatialOrder));
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
        writeMatrix(stream, node.interfaceSystem);
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
    stream.read(reinterpret_cast<char*>(&hierarchy.mFaceSpatialOrder), sizeof(hierarchy.mFaceSpatialOrder));
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
        node.interfaceSystem = readMatrix(stream);
        node.residual = readMatrix(stream);
    }
    require(bool(stream), "HST-R hierarchy file is truncated.");
    return hierarchy;
}

DenseMatrix makeGridBoundaryTransport(uint32_t cellsPerAxis, const std::vector<DenseMatrix>& cellTransport)
{
    require(cellsPerAxis > 0, "HST-R transport grid dimensions must be non-zero.");
    const uint3 dims(cellsPerAxis);
    const size_t cellCount = size_t(cellsPerAxis) * cellsPerAxis * cellsPerAxis;
    require(cellTransport.size() == cellCount, "HST-R transport grid operator count does not match its dimensions.");
    for (const DenseMatrix& transport : cellTransport)
        require(transport.rows() == kFaceCount && transport.cols() == kFaceCount, "HST-R transport grid cells must use six-direction operators.");

    std::vector<int32_t> externalPort(cellCount * kFaceCount, -1);
    uint32_t externalCount = 0;
    for (uint32_t face = 0; face < kFaceCount; ++face)
    {
        const uint32_t faceAxis = face / 2;
        uint32_t tangents[2];
        uint32_t tangentCount = 0;
        for (uint32_t axis = 0; axis < 3; ++axis)
            if (axis != faceAxis)
                tangents[tangentCount++] = axis;
        for (uint32_t v = 0; v < cellsPerAxis; ++v)
            for (uint32_t u = 0; u < cellsPerAxis; ++u)
            {
                uint3 p(0);
                (&p.x)[faceAxis] = (face & 1u) ? cellsPerAxis - 1 : 0;
                (&p.x)[tangents[0]] = u;
                (&p.x)[tangents[1]] = v;
                externalPort[kFaceCount * linearIndex(p, dims) + face] = int32_t(externalCount++);
            }
    }

    std::vector<int32_t> internalPort(cellCount * kFaceCount, -1);
    uint32_t internalCount = 0;
    for (size_t port = 0; port < internalPort.size(); ++port)
        if (externalPort[port] < 0)
            internalPort[port] = int32_t(internalCount++);

    DenseMatrix system(internalCount, internalCount);
    DenseMatrix rhs(internalCount, externalCount);
    for (uint32_t z = 0; z < cellsPerAxis; ++z)
        for (uint32_t y = 0; y < cellsPerAxis; ++y)
            for (uint32_t x = 0; x < cellsPerAxis; ++x)
            {
                const uint3 p(x, y, z);
                const size_t cell = linearIndex(p, dims);
                for (uint32_t face = 0; face < kFaceCount; ++face)
                {
                    const size_t port = kFaceCount * cell + face;
                    if (internalPort[port] < 0)
                        continue;
                    const uint32_t row = uint32_t(internalPort[port]);
                    system(row, row) = 1.f;
                    uint3 neighborPosition = p;
                    (&neighborPosition.x)[face / 2] += (face & 1u) ? 1 : -1;
                    const size_t neighbor = linearIndex(neighborPosition, dims);
                    const uint32_t outgoingFace = face ^ 1u;
                    for (uint32_t inputFace = 0; inputFace < kFaceCount; ++inputFace)
                    {
                        const float value = cellTransport[neighbor](outgoingFace, inputFace);
                        const size_t inputPort = kFaceCount * neighbor + inputFace;
                        if (internalPort[inputPort] >= 0)
                            system(row, uint32_t(internalPort[inputPort])) -= value;
                        else
                            rhs(row, uint32_t(externalPort[inputPort])) += value;
                    }
                }
            }
    const DenseMatrix internalResponse = internalCount > 0 ? solve(system, rhs) : DenseMatrix(0, externalCount);

    DenseMatrix result(externalCount, externalCount);
    for (size_t cell = 0; cell < cellCount; ++cell)
        for (uint32_t face = 0; face < kFaceCount; ++face)
        {
            const size_t port = kFaceCount * cell + face;
            if (externalPort[port] < 0)
                continue;
            const uint32_t row = uint32_t(externalPort[port]);
            for (uint32_t inputFace = 0; inputFace < kFaceCount; ++inputFace)
            {
                const float value = cellTransport[cell](face, inputFace);
                const size_t inputPort = kFaceCount * cell + inputFace;
                if (externalPort[inputPort] >= 0)
                    result(row, uint32_t(externalPort[inputPort])) += value;
                else
                    for (uint32_t col = 0; col < externalCount; ++col)
                        result(row, col) += value * internalResponse(uint32_t(internalPort[inputPort]), col);
            }
        }
    return result;
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
