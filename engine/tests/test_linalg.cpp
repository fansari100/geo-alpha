#include "geoalpha/state_space.hpp"

#include <gtest/gtest.h>

using namespace geoalpha;

TEST(Linalg, MatmulIdentity) {
    Matrix I = Matrix::identity(3);
    Matrix A(3, 3);
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 0) = 4; A(1, 1) = 5; A(1, 2) = 6;
    A(2, 0) = 7; A(2, 1) = 8; A(2, 2) = 9;
    Matrix C = linalg::matmul(I, A);
    for (std::size_t i = 0; i < 9; ++i) EXPECT_DOUBLE_EQ(C.data[i], A.data[i]);
}

TEST(Linalg, CholeskyRoundtrip) {
    // 2x2 SPD matrix.
    Matrix A(2, 2);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 5;
    Matrix L = A;
    ASSERT_TRUE(linalg::cholesky_lower(L));
    Matrix Lt = linalg::transpose(L);
    Matrix back = linalg::matmul(L, Lt);
    for (std::size_t i = 0; i < 4; ++i)
        EXPECT_NEAR(back.data[i], A.data[i], 1e-9);
}

TEST(Linalg, InvertSpdRoundtrip) {
    Matrix A(2, 2);
    A(0, 0) = 4; A(0, 1) = 2;
    A(1, 0) = 2; A(1, 1) = 5;
    Matrix Ainv = linalg::invert_spd(A);
    Matrix I = linalg::matmul(A, Ainv);
    EXPECT_NEAR(I(0, 0), 1.0, 1e-9);
    EXPECT_NEAR(I(0, 1), 0.0, 1e-9);
    EXPECT_NEAR(I(1, 0), 0.0, 1e-9);
    EXPECT_NEAR(I(1, 1), 1.0, 1e-9);
}
