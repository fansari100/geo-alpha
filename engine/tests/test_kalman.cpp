#include "geoalpha/state_space.hpp"

#include <gtest/gtest.h>

using namespace geoalpha;

TEST(Kalman, ConstantPositionConverges) {
    KalmanFilter kf(1, 1);
    Matrix F(1, 1); F(0, 0) = 1.0;
    Matrix H(1, 1); H(0, 0) = 1.0;
    Matrix Q(1, 1); Q(0, 0) = 1e-4;
    Matrix R(1, 1); R(0, 0) = 0.04;
    Matrix x0(1, 1); x0(0, 0) = 0.0;
    Matrix P0(1, 1); P0(0, 0) = 1.0;
    kf.set_transition(F);
    kf.set_observation(H);
    kf.set_process_noise(Q);
    kf.set_measurement_noise(R);
    kf.set_state(x0, P0);

    const double truth = 5.0;
    for (int i = 0; i < 100; ++i) {
        kf.predict();
        Matrix z(1, 1); z(0, 0) = truth + ((i % 2) ? 0.05 : -0.05);
        kf.update(z);
    }
    EXPECT_NEAR(kf.state()(0, 0), truth, 0.05);
    EXPECT_LT(kf.covariance()(0, 0), 0.01);
}

TEST(Kalman, ConstantVelocityTracking) {
    KalmanFilter kf(2, 1);
    const double dt = 0.1;
    Matrix F(2, 2); F(0, 0) = 1; F(0, 1) = dt; F(1, 0) = 0; F(1, 1) = 1;
    Matrix H(1, 2); H(0, 0) = 1; H(0, 1) = 0;
    Matrix Q = Matrix::from_diagonal({1e-4, 1e-4});
    Matrix R(1, 1); R(0, 0) = 0.04;
    Matrix x0(2, 1); x0(0, 0) = 0; x0(1, 0) = 0;
    Matrix P0 = Matrix::from_diagonal({1.0, 1.0});
    kf.set_transition(F);
    kf.set_observation(H);
    kf.set_process_noise(Q);
    kf.set_measurement_noise(R);
    kf.set_state(x0, P0);

    const double v = 2.0;
    double pos = 0.0;
    for (int i = 0; i < 200; ++i) {
        pos += v * dt;
        kf.predict();
        Matrix z(1, 1); z(0, 0) = pos;
        kf.update(z);
    }
    EXPECT_NEAR(kf.state()(0, 0), pos, 0.5);
    EXPECT_NEAR(kf.state()(1, 0), v, 0.2);
}
