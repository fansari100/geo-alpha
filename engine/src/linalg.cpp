// Tiny linear algebra core - just enough for Kalman / UKF.
//
// I deliberately avoid Eigen here.  This whole engine compiles cleanly
// on a stock g++ in under three seconds; pulling Eigen in would push
// the per-translation-unit times into the tens of seconds territory
// for what is, in the end, a fairly small set of dense ops on tiny
// matrices (state dim <= 16 in every realistic ISR target model).

#include "geoalpha/state_space.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace geoalpha {

Matrix Matrix::identity(std::size_t n) {
    Matrix M(n, n);
    for (std::size_t i = 0; i < n; ++i) M(i, i) = 1.0;
    return M;
}

Matrix Matrix::from_diagonal(const std::vector<double>& diag) {
    Matrix M(diag.size(), diag.size());
    for (std::size_t i = 0; i < diag.size(); ++i) M(i, i) = diag[i];
    return M;
}

namespace linalg {

Matrix matmul(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) throw std::invalid_argument("matmul: shape mismatch");
    Matrix C(A.rows, B.cols);
    for (std::size_t i = 0; i < A.rows; ++i) {
        for (std::size_t k = 0; k < A.cols; ++k) {
            const double a = A(i, k);
            for (std::size_t j = 0; j < B.cols; ++j) {
                C(i, j) += a * B(k, j);
            }
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix T(A.cols, A.rows);
    for (std::size_t i = 0; i < A.rows; ++i)
        for (std::size_t j = 0; j < A.cols; ++j)
            T(j, i) = A(i, j);
    return T;
}

Matrix add(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::invalid_argument("add: shape mismatch");
    Matrix C(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i) C.data[i] = A.data[i] + B.data[i];
    return C;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::invalid_argument("sub: shape mismatch");
    Matrix C(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i) C.data[i] = A.data[i] - B.data[i];
    return C;
}

Matrix scale(const Matrix& A, double s) {
    Matrix C(A.rows, A.cols);
    for (std::size_t i = 0; i < A.data.size(); ++i) C.data[i] = A.data[i] * s;
    return C;
}

bool cholesky_lower(Matrix& A) {
    if (A.rows != A.cols) return false;
    const std::size_t n = A.rows;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double s = A(i, j);
            for (std::size_t k = 0; k < j; ++k) s -= A(i, k) * A(j, k);
            if (i == j) {
                if (s <= 0.0) return false;
                A(i, i) = std::sqrt(s);
            } else {
                A(i, j) = s / A(j, j);
            }
        }
        for (std::size_t j = i + 1; j < n; ++j) A(i, j) = 0.0;
    }
    return true;
}

Matrix cholesky_solve(const Matrix& L, const Matrix& B) {
    const std::size_t n = L.rows;
    if (B.rows != n) throw std::invalid_argument("cholesky_solve: shape mismatch");
    Matrix Y(n, B.cols);
    // Forward substitution: L * Y = B
    for (std::size_t col = 0; col < B.cols; ++col) {
        for (std::size_t i = 0; i < n; ++i) {
            double s = B(i, col);
            for (std::size_t k = 0; k < i; ++k) s -= L(i, k) * Y(k, col);
            Y(i, col) = s / L(i, i);
        }
    }
    Matrix X(n, B.cols);
    // Back substitution: L^T * X = Y
    for (std::size_t col = 0; col < B.cols; ++col) {
        for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n) - 1; i >= 0; --i) {
            double s = Y(i, col);
            for (std::size_t k = static_cast<std::size_t>(i) + 1; k < n; ++k)
                s -= L(k, i) * X(k, col);
            X(i, col) = s / L(i, i);
        }
    }
    return X;
}

Matrix invert_spd(const Matrix& A) {
    Matrix L = A;
    if (!cholesky_lower(L)) throw std::runtime_error("invert_spd: matrix not PD");
    Matrix I = Matrix::identity(A.rows);
    return cholesky_solve(L, I);
}

}  // namespace linalg

}  // namespace geoalpha
