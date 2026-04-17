#include "geoalpha/state_space.hpp"

#include <gtest/gtest.h>

#include <cmath>

using namespace geoalpha;

namespace {
void identity_transition(const double* x_in, double* x_out, void*) {
    x_out[0] = x_in[0];
    x_out[1] = x_in[1];
}
void range_observation(const double* x, double* z_out, void*) {
    z_out[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
}
}  // namespace

TEST(UKF, NonlinearRangeObservationConverges) {
    UnscentedKalmanFilter ukf(2, 1);
    ukf.set_transition(identity_transition, nullptr);
    ukf.set_observation(range_observation, nullptr);

    Matrix Q = Matrix::from_diagonal({1e-3, 1e-3});
    Matrix R(1, 1); R(0, 0) = 0.05;
    ukf.set_process_noise(Q);
    ukf.set_measurement_noise(R);
    Matrix x0(2, 1); x0(0, 0) = 1.0; x0(1, 0) = 1.0;
    Matrix P0 = Matrix::from_diagonal({4.0, 4.0});
    ukf.set_state(x0, P0);

    const double truth_x = 3.0;
    const double truth_y = 4.0;
    const double truth_r = 5.0;
    for (int i = 0; i < 200; ++i) {
        ukf.predict();
        Matrix z(1, 1); z(0, 0) = truth_r;
        ukf.update(z);
    }
    const double r = std::sqrt(ukf.state()(0, 0) * ukf.state()(0, 0)
                             + ukf.state()(1, 0) * ukf.state()(1, 0));
    EXPECT_NEAR(r, truth_r, 0.2);
    (void)truth_x; (void)truth_y;  // unused except for documentation
}
