/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "Testing/UnitTest.h"
#include "Rendering/Volumes/HSTRHierarchy.h"
#include "Rendering/Volumes/HSTROperator.h"

#include <cmath>
#include <filesystem>

namespace Falcor
{
namespace
{
using namespace hstr;

void expectMatrixNear(CPUUnitTestContext& ctx, const DenseMatrix& actual, const DenseMatrix& expected, float tolerance = 1e-5f)
{
    EXPECT_EQ(actual.rows(), expected.rows());
    EXPECT_EQ(actual.cols(), expected.cols());
    for (size_t row = 0; row < actual.rows(); ++row)
        for (size_t col = 0; col < actual.cols(); ++col)
            EXPECT_LE(std::abs(actual(row, col) - expected(row, col)), tolerance) << "row=" << row << " col=" << col;
}
} // namespace

CPU_TEST(HSTRSchurComplement)
{
    const DenseMatrix system(3, 3, {4.f, 1.f, 2.f, 1.f, 3.f, 1.f, 2.f, 1.f, 5.f});
    const DenseMatrix reduced = schurComplement(system, 2);
    expectMatrixNear(ctx, reduced, DenseMatrix(2, 2, {3.2f, 0.6f, 0.6f, 2.8f}));

    const DenseMatrix rhs(3, 1, {1.f, 2.f, 3.f});
    const DenseMatrix fullSolution = solve(system, rhs);
    const DenseMatrix reducedRhs(2, 1, {1.f - 2.f * 3.f / 5.f, 2.f - 3.f / 5.f});
    const DenseMatrix reducedSolution = solve(reduced, reducedRhs);
    EXPECT_LE(std::abs(fullSolution(0, 0) - reducedSolution(0, 0)), 1e-5f);
    EXPECT_LE(std::abs(fullSolution(1, 0) - reducedSolution(1, 0)), 1e-5f);
}

CPU_TEST(HSTRResidualAndConservativeTrace)
{
    const TraceTransfer transfer = makeConservativeTraceTransfer(2, 2);
    expectMatrixNear(ctx, multiply(transfer.restriction, transfer.prolongation), DenseMatrix::identity(2));

    const DenseMatrix parent(2, 2, {0.8f, 0.1f, 0.2f, 0.7f});
    const DenseMatrix child(
        4,
        4,
        {
            0.82f,
            0.01f,
            0.10f,
            0.00f,
            0.02f,
            0.79f,
            0.00f,
            0.11f,
            0.20f,
            0.00f,
            0.72f,
            0.01f,
            0.00f,
            0.21f,
            0.02f,
            0.69f,
        }
    );
    const DenseMatrix correction = residual(child, parent, transfer.prolongation, transfer.restriction);
    const DenseMatrix reconstructed = subtract(child, correction);
    expectMatrixNear(ctx, reconstructed, multiply(multiply(transfer.prolongation, parent), transfer.restriction));
}

CPU_TEST(HSTRExactLowRankUpdate)
{
    const DenseMatrix base(3, 3, {4.f, 1.f, 0.f, 1.f, 3.f, 1.f, 0.f, 1.f, 2.f});
    const DenseMatrix u(3, 1, {0.5f, -0.25f, 0.75f});
    const DenseMatrix v(3, 1, {0.2f, 0.4f, -0.1f});
    const DenseMatrix rhs(3, 1, {1.f, 2.f, 3.f});

    DenseMatrix updated = base;
    const DenseMatrix update = multiply(u, transpose(v));
    for (size_t row = 0; row < updated.rows(); ++row)
        for (size_t col = 0; col < updated.cols(); ++col)
            updated(row, col) += update(row, col);

    expectMatrixNear(ctx, woodburySolve(base, u, v, rhs), solve(updated, rhs), 2e-5f);
}

CPU_TEST(HSTRGoalOrderedProgressiveTransport)
{
    const std::vector<ResidualAtom> atoms = {
        {0, 1.f, 100, 0.f},
        {1, 5.f, 100, 0.f},
        {2, 2.f, 50, 0.f},
        {3, 0.f, 1, 0.f},
    };
    const auto selected = selectResidualAtoms(atoms, 150, 1.f);
    EXPECT_EQ(selected.size(), 2);
    EXPECT_EQ(selected[0], 1);
    EXPECT_EQ(selected[1], 2);
}

CPU_TEST(HSTRHierarchyCompositionAndSerialization)
{
    const DenseMatrix leaf = makeLeafTransport(float3(0.4f, 0.6f, 0.8f), 0.98f, 0.75f, 0.4f);
    const Hierarchy hierarchy = Hierarchy::compile(uint3(2, 1, 1), {leaf, leaf});
    EXPECT_EQ(hierarchy.getNodes().size(), 3);
    EXPECT_LE(hierarchy.getNodes()[hierarchy.getRoot()].conservationError, 1e-5f);

    DenseMatrix incident(6, 3);
    incident(0, 0) = 4.f;
    incident(0, 1) = 3.f;
    incident(0, 2) = 2.f;
    const auto radiance = hierarchy.solve(incident);
    EXPECT_EQ(radiance.size(), 2);
    EXPECT_GT(radiance[0].x, 0.f);
    EXPECT_GT(radiance[1].x, 0.f);

    const auto path = std::filesystem::temp_directory_path() / "falcor_hstr_roundtrip.bin";
    hierarchy.save(path);
    const Hierarchy restored = Hierarchy::load(path);
    std::filesystem::remove(path);
    expectMatrixNear(ctx, restored.getNodes()[restored.getRoot()].transport, hierarchy.getNodes()[hierarchy.getRoot()].transport);
    const auto restoredRadiance = restored.solve(incident);
    EXPECT_LE(length(restoredRadiance[0] - radiance[0]), 1e-5f);
    EXPECT_LE(length(restoredRadiance[1] - radiance[1]), 1e-5f);
}

CPU_TEST(HSTRAdjointIdentity)
{
    const DenseMatrix leaf = makeLeafTransport(float3(0.5f), 0.95f, 0.8f, 0.5f);
    const Hierarchy hierarchy = Hierarchy::compile(uint3(1), {leaf});
    const DenseMatrix x(6, 1, {1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
    const DenseMatrix g(6, 1, {2.f, 1.f, 0.5f, 3.f, 1.5f, 4.f});
    const DenseMatrix sx = multiply(hierarchy.getNodes()[hierarchy.getRoot()].transport, x);
    const DenseMatrix stg = hierarchy.solveAdjoint(g);
    const DenseMatrix lhs = multiply(transpose(g), sx);
    const DenseMatrix rhs = multiply(transpose(stg), x);
    EXPECT_LE(std::abs(lhs(0, 0) - rhs(0, 0)), 1e-5f);
}

CPU_TEST(HSTRAdjointOrderedResidualAtoms)
{
    const DenseMatrix a = makeLeafTransport(float3(0.2f, 0.4f, 0.7f), 0.98f, 0.8f, 0.6f);
    const DenseMatrix b = makeLeafTransport(float3(1.1f, 0.5f, 0.3f), 0.98f, 0.8f, 0.6f);
    const Hierarchy hierarchy = Hierarchy::compile(uint3(2, 1, 1), {a, b});
    const DenseMatrix incident(6, 1, {1.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    const DenseMatrix goal(6, 1, {0.f, 1.f, 0.f, 0.f, 0.f, 0.f});
    const auto atoms = hierarchy.rankResidualAtoms(incident, goal);
    EXPECT_GT(atoms.size(), 0);
    for (size_t i = 1; i < atoms.size(); ++i)
        EXPECT_GE(atoms[i - 1].score, atoms[i].score);
}

CPU_TEST(HSTRLocalizedLeafRepair)
{
    const DenseMatrix clear = makeLeafTransport(float3(0.05f), 0.98f, 0.8f, 0.6f);
    const DenseMatrix dense = makeLeafTransport(float3(2.f), 0.98f, 0.8f, 0.6f);
    Hierarchy hierarchy = Hierarchy::compile(uint3(4, 2, 1), std::vector<DenseMatrix>(8, clear));
    const DenseMatrix before = hierarchy.getNodes()[hierarchy.getRoot()].transport;
    const uint32_t repaired = hierarchy.updateLeaf(3, dense);
    EXPECT_EQ(repaired, 3);
    const DenseMatrix after = hierarchy.getNodes()[hierarchy.getRoot()].transport;
    float change = 0.f;
    for (size_t row = 0; row < before.rows(); ++row)
        for (size_t col = 0; col < before.cols(); ++col)
            change += std::abs(after(row, col) - before(row, col));
    EXPECT_GT(change, 0.f);
}

CPU_TEST(HSTRMatrixFreeCompression)
{
    const DenseMatrix left(4, 2, {1.f, 0.f, 0.f, 2.f, 1.f, 1.f, -1.f, 0.5f});
    const DenseMatrix right(3, 2, {1.f, 0.f, 0.5f, 1.f, -1.f, 0.25f});
    const DenseMatrix matrix = multiply(left, transpose(right));
    const LowRankOperator compressed = compressOperator(
        4,
        3,
        [&matrix](const DenseMatrix& x) { return multiply(matrix, x); },
        [&matrix](const DenseMatrix& x) { return multiply(transpose(matrix), x); },
        1e-5f,
        3
    );
    EXPECT_EQ(compressed.rank(), 2);
    expectMatrixNear(ctx, compressed.reconstruct(), matrix, 2e-4f);
}

CPU_TEST(HSTRStaleFactorKrylovFallback)
{
    const DenseMatrix oldSystem(3, 3, {4.f, 1.f, 0.f, 1.f, 3.f, 1.f, 0.f, 1.f, 2.f});
    const DenseMatrix newSystem(3, 3, {4.4f, 0.8f, 0.2f, 1.1f, 3.5f, 0.7f, 0.1f, 1.3f, 2.6f});
    const DenseMatrix rhs(3, 1, {1.f, 2.f, 3.f});
    const KrylovResult result = gmres(
        3,
        [&newSystem](const DenseMatrix& x) { return multiply(newSystem, x); },
        [&oldSystem](const DenseMatrix& x) { return solve(oldSystem, x); },
        rhs,
        3,
        1e-6f
    );
    expectMatrixNear(ctx, result.solution, solve(newSystem, rhs), 2e-4f);
    EXPECT_LE(result.relativeResidual, 2e-4f);
}

CPU_TEST(HSTRResidualBoundsReachLeaves)
{
    const DenseMatrix a = makeLeafTransport(float3(0.1f), 0.98f, 0.8f, 0.6f);
    const DenseMatrix b = makeLeafTransport(float3(1.5f), 0.98f, 0.8f, 0.6f);
    const Hierarchy hierarchy = Hierarchy::compile(uint3(2, 1, 1), {a, b});
    const auto bounds = hierarchy.getLeafResidualBounds();
    EXPECT_EQ(bounds.size(), 2);
    EXPECT_GT(bounds[0], 0.f);
    EXPECT_EQ(bounds[0], bounds[1]);
}

CPU_TEST(HSTRPackedLeafSubstitutionMatchesSolve)
{
    const DenseMatrix a = makeLeafTransport(float3(0.2f, 0.5f, 0.9f), 0.98f, 0.8f, 0.6f);
    const DenseMatrix b = makeLeafTransport(float3(1.2f, 0.4f, 0.3f), 0.98f, 0.8f, 0.6f);
    const Hierarchy hierarchy = Hierarchy::compile(uint3(2, 1, 1), {a, b});
    const DenseMatrix incident(6, 3, {1.f, 2.f, 3.f, 0.f, 0.f, 0.f, 0.5f, 0.2f, 0.1f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f});
    const auto expected = hierarchy.solveFaces(incident);
    const auto transfer = hierarchy.getLeafTransferMatrices();
    const auto transport = hierarchy.getLeafTransportMatrices();
    for (size_t leaf = 0; leaf < transfer.size(); ++leaf)
    {
        const DenseMatrix outgoing = multiply(transport[leaf], multiply(transfer[leaf], incident));
        for (size_t face = 0; face < 6; ++face)
            for (size_t channel = 0; channel < 3; ++channel)
                EXPECT_LE(std::abs(outgoing(face, channel) - expected[6 * leaf + face][channel]), 1e-5f);
    }
}

CPU_TEST(HSTRThickLeafUsesDiffusionLimit)
{
    const DenseMatrix thin = makeLeafTransport(float3(0.2f), 1.f, 0.9f, 0.f);
    const DenseMatrix thick = makeLeafTransport(float3(20.f), 1.f, 0.9f, 0.f);
    float thinSpread = 0.f;
    float thickSpread = 0.f;
    for (size_t row = 0; row < 6; ++row)
    {
        thinSpread += std::abs(thin(row, 0) - 1.f / 6.f);
        thickSpread += std::abs(thick(row, 0) - 1.f / 6.f);
    }
    EXPECT_LT(thickSpread, thinSpread);
    float columnSum = 0.f;
    for (size_t row = 0; row < 6; ++row)
        columnSum += thick(row, 0);
    EXPECT_LE(columnSum, 1.f + 1e-5f);
}
} // namespace Falcor
