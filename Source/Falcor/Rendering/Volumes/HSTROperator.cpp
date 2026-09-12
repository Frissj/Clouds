/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#include "HSTROperator.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Falcor::hstr
{
namespace
{
void require(bool condition, const char* message)
{
    if (!condition)
        FALCOR_THROW(message);
}
} // namespace

DenseMatrix::DenseMatrix(size_t rows, size_t cols, float value) : mRows(rows), mCols(cols), mData(rows * cols, value) {}

DenseMatrix::DenseMatrix(size_t rows, size_t cols, std::initializer_list<float> values) : mRows(rows), mCols(cols), mData(values)
{
    require(values.size() == rows * cols, "DenseMatrix initializer has the wrong size.");
}

DenseMatrix DenseMatrix::identity(size_t size)
{
    DenseMatrix result(size, size);
    for (size_t i = 0; i < size; ++i)
        result(i, i) = 1.f;
    return result;
}

DenseMatrix transpose(const DenseMatrix& matrix)
{
    DenseMatrix result(matrix.cols(), matrix.rows());
    for (size_t row = 0; row < matrix.rows(); ++row)
        for (size_t col = 0; col < matrix.cols(); ++col)
            result(col, row) = matrix(row, col);
    return result;
}

DenseMatrix multiply(const DenseMatrix& lhs, const DenseMatrix& rhs)
{
    require(lhs.cols() == rhs.rows(), "Dense matrix multiply dimension mismatch.");
    DenseMatrix result(lhs.rows(), rhs.cols());
    for (size_t row = 0; row < lhs.rows(); ++row)
        for (size_t k = 0; k < lhs.cols(); ++k)
            for (size_t col = 0; col < rhs.cols(); ++col)
                result(row, col) += lhs(row, k) * rhs(k, col);
    return result;
}

DenseMatrix subtract(const DenseMatrix& lhs, const DenseMatrix& rhs)
{
    require(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols(), "Dense matrix subtract dimension mismatch.");
    DenseMatrix result(lhs.rows(), lhs.cols());
    for (size_t row = 0; row < lhs.rows(); ++row)
        for (size_t col = 0; col < lhs.cols(); ++col)
            result(row, col) = lhs(row, col) - rhs(row, col);
    return result;
}

DenseMatrix solve(DenseMatrix matrix, DenseMatrix rhs)
{
    require(matrix.rows() == matrix.cols(), "Dense solve requires a square matrix.");
    require(matrix.rows() == rhs.rows(), "Dense solve right-hand side dimension mismatch.");

    const size_t n = matrix.rows();
    for (size_t pivot = 0; pivot < n; ++pivot)
    {
        size_t best = pivot;
        for (size_t row = pivot + 1; row < n; ++row)
            if (std::abs(matrix(row, pivot)) > std::abs(matrix(best, pivot)))
                best = row;

        require(std::abs(matrix(best, pivot)) > 64.f * std::numeric_limits<float>::epsilon(), "Dense solve matrix is singular.");
        if (best != pivot)
        {
            for (size_t col = 0; col < n; ++col)
                std::swap(matrix(pivot, col), matrix(best, col));
            for (size_t col = 0; col < rhs.cols(); ++col)
                std::swap(rhs(pivot, col), rhs(best, col));
        }

        const float diagonal = matrix(pivot, pivot);
        for (size_t col = pivot; col < n; ++col)
            matrix(pivot, col) /= diagonal;
        for (size_t col = 0; col < rhs.cols(); ++col)
            rhs(pivot, col) /= diagonal;

        for (size_t row = 0; row < n; ++row)
        {
            if (row == pivot)
                continue;
            const float factor = matrix(row, pivot);
            for (size_t col = pivot; col < n; ++col)
                matrix(row, col) -= factor * matrix(pivot, col);
            for (size_t col = 0; col < rhs.cols(); ++col)
                rhs(row, col) -= factor * rhs(pivot, col);
        }
    }
    return rhs;
}

DenseMatrix LowRankOperator::apply(const DenseMatrix& rhs) const
{
    if (rank() == 0)
        return DenseMatrix(left.rows(), rhs.cols());
    return multiply(left, multiply(transpose(right), rhs));
}

DenseMatrix LowRankOperator::reconstruct() const
{
    return rank() == 0 ? DenseMatrix(left.rows(), right.rows()) : multiply(left, transpose(right));
}

LowRankOperator compressOperator(
    size_t rows,
    size_t cols,
    const MatrixApply& apply,
    const MatrixApply& applyAdjoint,
    float relativeTolerance,
    size_t maxRank
)
{
    require(rows > 0 && cols > 0, "Low-rank operator dimensions must be non-zero.");
    require(relativeTolerance >= 0.f, "Low-rank operator tolerance must be non-negative.");
    maxRank = std::min({maxRank, rows, cols});

    std::vector<std::vector<float>> leftColumns;
    std::vector<std::vector<float>> rightColumns;
    float firstSigma = 0.f;
    float lastSigma = 0.f;
    for (size_t rank = 0; rank < maxRank; ++rank)
    {
        DenseMatrix v(cols, 1);
        for (size_t i = 0; i < cols; ++i)
            v(i, 0) = std::sin(float((i + 1) * (rank + 1)) * 1.6180339f);

        DenseMatrix u;
        for (uint32_t iteration = 0; iteration < 16; ++iteration)
        {
            u = apply(v);
            for (size_t k = 0; k < leftColumns.size(); ++k)
            {
                float projection = 0.f;
                for (size_t i = 0; i < cols; ++i)
                    projection += rightColumns[k][i] * v(i, 0);
                for (size_t i = 0; i < rows; ++i)
                    u(i, 0) -= leftColumns[k][i] * projection;
            }
            v = applyAdjoint(u);
            for (size_t k = 0; k < rightColumns.size(); ++k)
            {
                float projection = 0.f;
                for (size_t i = 0; i < rows; ++i)
                    projection += leftColumns[k][i] * u(i, 0);
                for (size_t i = 0; i < cols; ++i)
                    v(i, 0) -= rightColumns[k][i] * projection;
            }
            float norm = 0.f;
            for (size_t i = 0; i < cols; ++i)
                norm += v(i, 0) * v(i, 0);
            norm = std::sqrt(norm);
            if (norm <= 1e-12f)
                break;
            for (size_t i = 0; i < cols; ++i)
                v(i, 0) /= norm;
        }

        u = apply(v);
        for (size_t k = 0; k < leftColumns.size(); ++k)
        {
            float projection = 0.f;
            for (size_t i = 0; i < cols; ++i)
                projection += rightColumns[k][i] * v(i, 0);
            for (size_t i = 0; i < rows; ++i)
                u(i, 0) -= leftColumns[k][i] * projection;
        }
        float sigma = 0.f;
        for (size_t i = 0; i < rows; ++i)
            sigma += u(i, 0) * u(i, 0);
        sigma = std::sqrt(sigma);
        if (rank == 0)
            firstSigma = sigma;
        if (sigma <= std::max(1e-7f, relativeTolerance * firstSigma))
        {
            lastSigma = sigma;
            break;
        }

        std::vector<float> left(rows);
        std::vector<float> right(cols);
        for (size_t i = 0; i < rows; ++i)
            left[i] = u(i, 0);
        for (size_t i = 0; i < cols; ++i)
            right[i] = v(i, 0);
        leftColumns.push_back(std::move(left));
        rightColumns.push_back(std::move(right));
        lastSigma = sigma;
    }

    LowRankOperator result;
    result.left = DenseMatrix(rows, leftColumns.size());
    result.right = DenseMatrix(cols, rightColumns.size());
    for (size_t k = 0; k < leftColumns.size(); ++k)
    {
        for (size_t i = 0; i < rows; ++i)
            result.left(i, k) = leftColumns[k][i];
        for (size_t i = 0; i < cols; ++i)
            result.right(i, k) = rightColumns[k][i];
    }
    result.residualNorm = lastSigma;
    return result;
}

LowRankOperator compress(const DenseMatrix& matrix, float relativeTolerance, size_t maxRank)
{
    return compressOperator(
        matrix.rows(),
        matrix.cols(),
        [&matrix](const DenseMatrix& rhs) { return multiply(matrix, rhs); },
        [&matrix](const DenseMatrix& rhs) { return multiply(transpose(matrix), rhs); },
        relativeTolerance,
        maxRank
    );
}

KrylovResult gmres(
    size_t dimension,
    const MatrixApply& apply,
    const MatrixApply& precondition,
    const DenseMatrix& rhs,
    uint32_t maxIterations,
    float tolerance
)
{
    require(rhs.rows() == dimension && rhs.cols() == 1, "HST-R GMRES currently accepts one right-hand side.");
    require(maxIterations > 0, "HST-R GMRES iteration count must be non-zero.");
    require(tolerance >= 0.f, "HST-R GMRES tolerance must be non-negative.");
    const uint32_t count = std::min<uint32_t>(maxIterations, uint32_t(dimension));
    auto vectorNorm = [](const DenseMatrix& v)
    {
        float value = 0.f;
        for (float x : v.data())
            value += x * x;
        return std::sqrt(value);
    };

    const DenseMatrix initial = precondition(rhs);
    const float beta = vectorNorm(initial);
    if (beta <= 1e-20f)
        return {DenseMatrix(dimension, 1), 0, 0.f};

    std::vector<DenseMatrix> basis;
    DenseMatrix first = initial;
    for (size_t row = 0; row < dimension; ++row)
        first(row, 0) /= beta;
    basis.push_back(std::move(first));
    DenseMatrix hessenberg(count + 1, count);
    auto solveProjected = [&](uint32_t columns)
    {
        DenseMatrix normal(columns, columns);
        DenseMatrix projected(columns, 1);
        for (uint32_t row = 0; row < columns; ++row)
        {
            projected(row, 0) = beta * hessenberg(0, row);
            for (uint32_t col = 0; col < columns; ++col)
                for (uint32_t k = 0; k <= columns; ++k)
                    normal(row, col) += hessenberg(k, row) * hessenberg(k, col);
            normal(row, row) += 1e-7f;
        }
        return solve(normal, projected);
    };
    uint32_t built = 0;
    for (uint32_t col = 0; col < count; ++col)
    {
        DenseMatrix w = precondition(apply(basis[col]));
        for (uint32_t row = 0; row <= col; ++row)
        {
            float projection = 0.f;
            for (size_t i = 0; i < dimension; ++i)
                projection += basis[row](i, 0) * w(i, 0);
            hessenberg(row, col) = projection;
            for (size_t i = 0; i < dimension; ++i)
                w(i, 0) -= projection * basis[row](i, 0);
        }
        hessenberg(col + 1, col) = vectorNorm(w);
        built = col + 1;
        const DenseMatrix coefficients = solveProjected(built);
        float projectedResidual = 0.f;
        for (uint32_t row = 0; row <= built; ++row)
        {
            float value = row == 0 ? beta : 0.f;
            for (uint32_t k = 0; k < built; ++k)
                value -= hessenberg(row, k) * coefficients(k, 0);
            projectedResidual += value * value;
        }
        if (std::sqrt(projectedResidual) <= tolerance * beta || hessenberg(col + 1, col) <= 1e-7f || col + 1 == count)
            break;
        for (size_t i = 0; i < dimension; ++i)
            w(i, 0) /= hessenberg(col + 1, col);
        basis.push_back(std::move(w));
    }

    const DenseMatrix coefficients = solveProjected(built);
    DenseMatrix solution(dimension, 1);
    for (uint32_t col = 0; col < built; ++col)
        for (size_t row = 0; row < dimension; ++row)
            solution(row, 0) += basis[col](row, 0) * coefficients(col, 0);

    const DenseMatrix residualVector = subtract(rhs, apply(solution));
    return {solution, built, vectorNorm(residualVector) / std::max(1e-20f, vectorNorm(rhs))};
}

DenseMatrix schurComplement(const DenseMatrix& system, size_t exteriorDofs)
{
    require(system.rows() == system.cols(), "Schur complement requires a square system.");
    require(exteriorDofs > 0 && exteriorDofs < system.rows(), "Schur complement partition is invalid.");

    const size_t internalDofs = system.rows() - exteriorDofs;
    DenseMatrix aEE(exteriorDofs, exteriorDofs);
    DenseMatrix aEI(exteriorDofs, internalDofs);
    DenseMatrix aIE(internalDofs, exteriorDofs);
    DenseMatrix aII(internalDofs, internalDofs);
    for (size_t row = 0; row < system.rows(); ++row)
        for (size_t col = 0; col < system.cols(); ++col)
        {
            if (row < exteriorDofs && col < exteriorDofs)
                aEE(row, col) = system(row, col);
            else if (row < exteriorDofs)
                aEI(row, col - exteriorDofs) = system(row, col);
            else if (col < exteriorDofs)
                aIE(row - exteriorDofs, col) = system(row, col);
            else
                aII(row - exteriorDofs, col - exteriorDofs) = system(row, col);
        }
    return subtract(aEE, multiply(aEI, solve(aII, aIE)));
}

DenseMatrix residual(const DenseMatrix& child, const DenseMatrix& parent, const DenseMatrix& prolongation, const DenseMatrix& restriction)
{
    return subtract(child, multiply(multiply(prolongation, parent), restriction));
}

DenseMatrix woodburySolve(const DenseMatrix& base, const DenseMatrix& u, const DenseMatrix& v, const DenseMatrix& rhs)
{
    require(base.rows() == base.cols(), "Woodbury base matrix must be square.");
    require(u.rows() == base.rows() && v.rows() == base.rows(), "Woodbury update dimension mismatch.");
    require(u.cols() == v.cols(), "Woodbury update factors must have equal rank.");
    require(rhs.rows() == base.rows(), "Woodbury right-hand side dimension mismatch.");

    const DenseMatrix y = solve(base, rhs);
    const DenseMatrix z = solve(base, u);
    const DenseMatrix vt = transpose(v);
    DenseMatrix middle = multiply(vt, z);
    for (size_t i = 0; i < middle.rows(); ++i)
        middle(i, i) += 1.f;
    return subtract(y, multiply(z, solve(middle, multiply(vt, y))));
}

TraceTransfer makeConservativeTraceTransfer(size_t coarseDofs, size_t refinement)
{
    require(coarseDofs > 0 && refinement > 0, "Trace transfer dimensions must be non-zero.");
    std::vector<float> coarseWeights(coarseDofs, 1.f);
    std::vector<float> fineWeights(coarseDofs * refinement, 1.f / float(refinement));
    std::vector<uint32_t> fineToCoarse(coarseDofs * refinement);
    for (size_t coarse = 0; coarse < coarseDofs; ++coarse)
        for (size_t child = 0; child < refinement; ++child)
            fineToCoarse[coarse * refinement + child] = uint32_t(coarse);
    return makeConservativeTraceTransfer(coarseWeights, fineWeights, fineToCoarse);
}

TraceTransfer makeConservativeTraceTransfer(
    const std::vector<float>& coarseWeights,
    const std::vector<float>& fineWeights,
    const std::vector<uint32_t>& fineToCoarse
)
{
    require(!coarseWeights.empty() && !fineWeights.empty(), "Trace transfer dimensions must be non-zero.");
    require(fineWeights.size() == fineToCoarse.size(), "Fine trace weights and assignments must have equal size.");

    TraceTransfer result{
        DenseMatrix(fineWeights.size(), coarseWeights.size()),
        DenseMatrix(coarseWeights.size(), fineWeights.size()),
    };
    std::vector<float> assignedWeights(coarseWeights.size(), 0.f);
    for (size_t fine = 0; fine < fineWeights.size(); ++fine)
    {
        const uint32_t coarse = fineToCoarse[fine];
        require(coarse < coarseWeights.size(), "Fine trace assignment is out of range.");
        require(fineWeights[fine] > 0.f, "Fine trace quadrature weights must be positive.");
        result.prolongation(fine, coarse) = 1.f;
        assignedWeights[coarse] += fineWeights[fine];
    }
    for (size_t coarse = 0; coarse < coarseWeights.size(); ++coarse)
    {
        require(coarseWeights[coarse] > 0.f, "Coarse trace quadrature weights must be positive.");
        const float tolerance = 1e-5f * std::max(coarseWeights[coarse], assignedWeights[coarse]);
        require(std::abs(coarseWeights[coarse] - assignedWeights[coarse]) <= tolerance, "Nested trace quadrature weights do not preserve flux.");
    }
    for (size_t fine = 0; fine < fineWeights.size(); ++fine)
    {
        const uint32_t coarse = fineToCoarse[fine];
        result.restriction(coarse, fine) = fineWeights[fine] / coarseWeights[coarse];
    }
    return result;
}

std::vector<uint32_t> selectResidualAtoms(
    const std::vector<ResidualAtom>& atoms,
    uint64_t byteBudget,
    float updateBudget,
    float updateCostToBytes
)
{
    std::vector<const ResidualAtom*> ranked;
    ranked.reserve(atoms.size());
    for (const ResidualAtom& atom : atoms)
        ranked.push_back(&atom);
    std::stable_sort(
        ranked.begin(),
        ranked.end(),
        [updateCostToBytes](const ResidualAtom* lhs, const ResidualAtom* rhs)
        {
            const float lhsCost = std::max(1.f, float(lhs->byteSize) + updateCostToBytes * lhs->updateCost);
            const float rhsCost = std::max(1.f, float(rhs->byteSize) + updateCostToBytes * rhs->updateCost);
            return lhs->goalErrorReduction / lhsCost > rhs->goalErrorReduction / rhsCost;
        }
    );

    uint64_t usedBytes = 0;
    float usedUpdate = 0.f;
    std::vector<uint32_t> selected;
    for (const ResidualAtom* atom : ranked)
    {
        if (atom->goalErrorReduction <= 0.f || atom->byteSize > byteBudget - usedBytes || atom->updateCost > updateBudget - usedUpdate)
            continue;
        selected.push_back(atom->index);
        usedBytes += atom->byteSize;
        usedUpdate += atom->updateCost;
    }
    return selected;
}
} // namespace Falcor::hstr
