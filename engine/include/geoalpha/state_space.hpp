// State-space estimators (Kalman / UKF / particle).
//
// This header defines the public C++ API for the engine.  It's
// deliberately stateless w.r.t. allocations - all working buffers live
// inside the filter objects so the per-step `predict`/`update` calls
// don't churn the heap.  That matters when you're processing a
// 100 Hz radar return or stepping through a million-particle filter.
//
// The same machinery is what powers the HMM regime detector on the
// Python side - linear-Gaussian state-space filter with a regime
// switch is just a Kalman bank.  Sharing this code keeps the
// statistical conventions consistent across the whole stack.
#pragma once

#include <cstddef>
#include <vector>

namespace geoalpha {

// Dynamically-sized matrix view backed by a flat std::vector<double>.
//
// Row-major layout to match numpy's default ordering across the
// pybind11 boundary.
struct Matrix {
    std::size_t rows{0};
    std::size_t cols{0};
    std::vector<double> data;

    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r * c, 0.0) {}

    [[nodiscard]] double& operator()(std::size_t i, std::size_t j) noexcept {
        return data[i * cols + j];
    }
    [[nodiscard]] double operator()(std::size_t i, std::size_t j) const noexcept {
        return data[i * cols + j];
    }

    static Matrix identity(std::size_t n);
    static Matrix from_diagonal(const std::vector<double>& diag);
};

// Linear matrix algebra primitives - kept tiny and dependency-free.
namespace linalg {

Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);
Matrix add(const Matrix& A, const Matrix& B);
Matrix sub(const Matrix& A, const Matrix& B);
Matrix scale(const Matrix& A, double s);

// In-place Cholesky factorisation - returns false on non-PD input.
bool cholesky_lower(Matrix& A);

// Solve A * X = B given A's lower Cholesky factor L.
Matrix cholesky_solve(const Matrix& L, const Matrix& B);

// Convenience: invert a symmetric positive-definite matrix via Cholesky.
Matrix invert_spd(const Matrix& A);

}  // namespace linalg


// --------------------------------------------------------------------- //
// Linear Kalman filter.
//
// State equation:  x_k = F x_{k-1} + B u_{k-1} + w,   w ~ N(0, Q)
// Obs equation:    z_k = H x_k             + v,   v ~ N(0, R)
//
// Use cases on the ISR side: simple constant-velocity target tracking
// from sensor reports, smoothing of integrated radiance time series,
// and as the per-regime kernel of a Hidden Markov filter bank.
// --------------------------------------------------------------------- //
class KalmanFilter {
public:
    KalmanFilter(std::size_t state_dim, std::size_t obs_dim);

    void set_transition(const Matrix& F);
    void set_control(const Matrix& B);
    void set_observation(const Matrix& H);
    void set_process_noise(const Matrix& Q);
    void set_measurement_noise(const Matrix& R);
    void set_state(const Matrix& x, const Matrix& P);

    void predict();
    void predict(const Matrix& u);            // with control input
    void update(const Matrix& z);             // measurement update

    [[nodiscard]] const Matrix& state() const noexcept { return x_; }
    [[nodiscard]] const Matrix& covariance() const noexcept { return P_; }
    [[nodiscard]] double last_innovation_norm() const noexcept { return last_inn_; }

private:
    std::size_t n_;
    std::size_t m_;
    Matrix F_, B_, H_, Q_, R_;
    Matrix x_, P_;
    double last_inn_{0.0};
};


// --------------------------------------------------------------------- //
// Unscented Kalman Filter for nonlinear state / measurement functions.
//
// Used when the ISR target dynamics aren't linear (e.g. coordinated
// turn for an aircraft, ballistic trajectory) - we still propagate
// Gaussian beliefs through the deterministic Julier/Uhlmann sigma
// points, which is more accurate than the Extended KF's local
// linearisation.
// --------------------------------------------------------------------- //
class UnscentedKalmanFilter {
public:
    using TransitionFn = void (*)(const double* x_in, double* x_out, void* ctx);
    using ObservationFn = void (*)(const double* x, double* z_out, void* ctx);

    UnscentedKalmanFilter(std::size_t state_dim, std::size_t obs_dim,
                          double alpha = 1e-3, double beta = 2.0, double kappa = 0.0);

    void set_state(const Matrix& x, const Matrix& P);
    void set_process_noise(const Matrix& Q);
    void set_measurement_noise(const Matrix& R);
    void set_transition(TransitionFn fn, void* ctx) { transition_ = fn; transition_ctx_ = ctx; }
    void set_observation(ObservationFn fn, void* ctx) { observation_ = fn; observation_ctx_ = ctx; }

    void predict();
    void update(const Matrix& z);

    [[nodiscard]] const Matrix& state() const noexcept { return x_; }
    [[nodiscard]] const Matrix& covariance() const noexcept { return P_; }

private:
    std::size_t n_, m_;
    double alpha_, beta_, kappa_;
    double lambda_;
    std::vector<double> wm_, wc_;
    Matrix x_, P_, Q_, R_;
    TransitionFn transition_{nullptr};
    void* transition_ctx_{nullptr};
    ObservationFn observation_{nullptr};
    void* observation_ctx_{nullptr};

    void generate_sigma_points(std::vector<Matrix>& sigmas) const;
};

}  // namespace geoalpha
