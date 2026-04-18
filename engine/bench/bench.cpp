// Latency micro-benchmark for the engine.
//
// Reports per-step nanoseconds for the linear KF and the UKF on
// realistic ISR target dimensionalities (state = 4 or 6).
//
//   geoalpha_bench
//   geoalpha_bench --steps 200000

#include "geoalpha/state_space.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

using clk = std::chrono::steady_clock;
using namespace geoalpha;

namespace {
void cv_transition(const double* x_in, double* x_out, void*) {
    constexpr double dt = 0.05;
    x_out[0] = x_in[0] + dt * x_in[2];
    x_out[1] = x_in[1] + dt * x_in[3];
    x_out[2] = x_in[2];
    x_out[3] = x_in[3];
}
void range_bearing(const double* x, double* z_out, void*) {
    z_out[0] = std::sqrt(x[0] * x[0] + x[1] * x[1]);
    z_out[1] = std::atan2(x[1], x[0]);
}
}  // namespace

int main(int argc, char** argv) {
    int steps = 50000;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--steps" && i + 1 < argc)
            steps = std::stoi(argv[++i]);
    }

    {
        const double dt = 0.05;
        KalmanFilter kf(4, 2);
        Matrix F(4, 4);
        F(0, 0) = 1; F(0, 2) = dt;
        F(1, 1) = 1; F(1, 3) = dt;
        F(2, 2) = 1; F(3, 3) = 1;
        Matrix H(2, 4); H(0, 0) = 1; H(1, 1) = 1;
        kf.set_transition(F);
        kf.set_observation(H);
        kf.set_process_noise(Matrix::from_diagonal({1e-4, 1e-4, 1e-3, 1e-3}));
        kf.set_measurement_noise(Matrix::from_diagonal({0.05, 0.05}));
        Matrix x0(4, 1);
        kf.set_state(x0, Matrix::from_diagonal({1, 1, 1, 1}));
        Matrix z(2, 1);

        auto t0 = clk::now();
        for (int i = 0; i < steps; ++i) {
            kf.predict();
            z(0, 0) = std::sin(i * 0.001);
            z(1, 0) = std::cos(i * 0.001);
            kf.update(z);
        }
        auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
        std::cout << "Linear KF (4x2): "
                  << static_cast<double>(dt_ns) / steps
                  << " ns/step  (" << steps << " iters)\n";
    }
    {
        UnscentedKalmanFilter ukf(4, 2);
        ukf.set_transition(cv_transition, nullptr);
        ukf.set_observation(range_bearing, nullptr);
        ukf.set_process_noise(Matrix::from_diagonal({1e-4, 1e-4, 1e-3, 1e-3}));
        ukf.set_measurement_noise(Matrix::from_diagonal({0.05, 0.005}));
        Matrix x0(4, 1); x0(0, 0) = 5; x0(1, 0) = 5;
        ukf.set_state(x0, Matrix::from_diagonal({1, 1, 1, 1}));
        Matrix z(2, 1);

        auto t0 = clk::now();
        for (int i = 0; i < steps; ++i) {
            ukf.predict();
            z(0, 0) = 7.07;
            z(1, 0) = 0.785;
            ukf.update(z);
        }
        auto dt_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0).count();
        std::cout << "UKF (4x2):       "
                  << static_cast<double>(dt_ns) / steps
                  << " ns/step  (" << steps << " iters)\n";
    }
    return 0;
}
