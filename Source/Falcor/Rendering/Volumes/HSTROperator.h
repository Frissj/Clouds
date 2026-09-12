/***************************************************************************
 # Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 **************************************************************************/
#pragma once

#include "Falcor.h"

#include <functional>

namespace Falcor::hstr
{
/** Small dense matrix used by the HST-R research compiler.
 *
 * This deliberately targets the small interface and update systems left after
 * skeletonization. Large transport operators stay matrix-free.
 */
class FALCOR_API DenseMatrix
{
public:
    DenseMatrix() = default;
    DenseMatrix(size_t rows, size_t cols, float value = 0.f);
    DenseMatrix(size_t rows, size_t cols, std::initializer_list<float> values);

    size_t rows() const { return mRows; }
    size_t cols() const { return mCols; }
    bool empty() const { return mData.empty(); }

    float& operator()(size_t row, size_t col) { return mData[row * mCols + col]; }
    float operator()(size_t row, size_t col) const { return mData[row * mCols + col]; }

    const std::vector<float>& data() const { return mData; }
    static DenseMatrix identity(size_t size);

private:
    size_t mRows = 0;
    size_t mCols = 0;
    std::vector<float> mData;
};

FALCOR_API DenseMatrix transpose(const DenseMatrix& matrix);
FALCOR_API DenseMatrix multiply(const DenseMatrix& lhs, const DenseMatrix& rhs);
FALCOR_API DenseMatrix subtract(const DenseMatrix& lhs, const DenseMatrix& rhs);
FALCOR_API DenseMatrix solve(DenseMatrix matrix, DenseMatrix rhs);

struct FALCOR_API LowRankOperator
{
    DenseMatrix left;
    DenseMatrix right;
    float residualNorm = 0.f;

    size_t rank() const { return left.cols(); }
    DenseMatrix apply(const DenseMatrix& rhs) const;
    DenseMatrix reconstruct() const;
};

using MatrixApply = std::function<DenseMatrix(const DenseMatrix&)>;

/** Randomized matrix-free compression using only A*x and A^T*x products. */
FALCOR_API LowRankOperator compressOperator(
    size_t rows,
    size_t cols,
    const MatrixApply& apply,
    const MatrixApply& applyAdjoint,
    float relativeTolerance,
    size_t maxRank
);
FALCOR_API LowRankOperator compress(const DenseMatrix& matrix, float relativeTolerance, size_t maxRank);

struct FALCOR_API KrylovResult
{
    DenseMatrix solution;
    uint32_t iterations = 0;
    float relativeResidual = 0.f;
};

/** Small left-preconditioned GMRES fallback for updates beyond Woodbury rank. */
FALCOR_API KrylovResult gmres(
    size_t dimension,
    const MatrixApply& apply,
    const MatrixApply& precondition,
    const DenseMatrix& rhs,
    uint32_t maxIterations,
    float tolerance
);

/** Eliminates the trailing internal degrees of freedom from a coupled system. */
FALCOR_API DenseMatrix schurComplement(const DenseMatrix& system, size_t exteriorDofs);

/** Computes D = child - P parent R for residual transport pages. */
FALCOR_API DenseMatrix
residual(const DenseMatrix& child, const DenseMatrix& parent, const DenseMatrix& prolongation, const DenseMatrix& restriction);

/** Exact Sherman-Morrison-Woodbury solve for (A + U V^T) x = b. */
FALCOR_API DenseMatrix woodburySolve(const DenseMatrix& base, const DenseMatrix& u, const DenseMatrix& v, const DenseMatrix& rhs);

struct FALCOR_API TraceTransfer
{
    DenseMatrix prolongation;
    DenseMatrix restriction;
};

/** Piecewise-constant conservative trace transfer with R P = I. */
FALCOR_API TraceTransfer makeConservativeTraceTransfer(size_t coarseDofs, size_t refinement);

struct FALCOR_API ResidualAtom
{
    uint32_t index = 0;
    float goalErrorReduction = 0.f;
    uint64_t byteSize = 0;
    float updateCost = 0.f;
};

/** Selects progressive correction atoms by goal-error reduction per cost. */
FALCOR_API std::vector<uint32_t> selectResidualAtoms(
    const std::vector<ResidualAtom>& atoms,
    uint64_t byteBudget,
    float updateBudget,
    float updateCostToBytes = 1.f
);
} // namespace Falcor::hstr
