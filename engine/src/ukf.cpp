// Unscented Kalman Filter (Julier & Uhlmann).
//
// We propagate 2n+1 sigma points through the nonlinear transition and
// observation functions, then reconstruct Gaussian beliefs from their
// weighted moments.  This sidesteps the linearisation error of the
// EKF for moderately nonlinear ISR target dynamics (coordinated turn,
// ballistic, etc).
//
// Convention: alpha controls the spread of the sigma points (small
// positive number, typically 1e-3), beta absorbs prior knowledge of
// distributional moments (2 is exact for Gaussian), kappa is a
// secondary scaling parameter (commonly 0 or 3 - n).

#include "geoalpha/state_space.hpp"

#include <cmath>
#include <stdexcept>

namespace geoalpha {

UnscentedKalmanFilter::UnscentedKalmanFilter(std::size_t state_dim, std::size_t obs_dim,
                                             double alpha, double beta, double kappa)
    : n_(state_dim), m_(obs_dim),
      alpha_(alpha), beta_(beta), kappa_(kappa),
      Q_(Matrix::identity(state_dim)),
      R_(Matrix::identity(obs_dim)),
      x_(state_dim, 1),
      P_(Matrix::identity(state_dim))
{
    lambda_ = alpha * alpha * (static_cast<double>(n_) + kappa) - static_cast<double>(n_);
    const std::size_t S = 2 * n_ + 1;
    wm_.resize(S);
    wc_.resize(S);
    wm_[0] = lambda_ / (n_ + lambda_);
    wc_[0] = wm_[0] + (1.0 - alpha * alpha + beta);
    const double w = 0.5 / (n_ + lambda_);
    for (std::size_t i = 1; i < S; ++i) {
        wm_[i] = w;
        wc_[i] = w;
    }
}

void UnscentedKalmanFilter::set_state(const Matrix& x, const Matrix& P) {
    if (x.rows != n_ || x.cols != 1) throw std::invalid_argument("state shape");
    if (P.rows != n_ || P.cols != n_) throw std::invalid_argument("covariance shape");
    x_ = x;
    P_ = P;
}

void UnscentedKalmanFilter::set_process_noise(const Matrix& Q)     { Q_ = Q; }
void UnscentedKalmanFilter::set_measurement_noise(const Matrix& R) { R_ = R; }

void UnscentedKalmanFilter::generate_sigma_points(std::vector<Matrix>& sigmas) const {
    using namespace linalg;
    sigmas.clear();
    sigmas.reserve(2 * n_ + 1);

    Matrix scaled = scale(P_, static_cast<double>(n_) + lambda_);
    Matrix L = scaled;
    if (!cholesky_lower(L)) {
        throw std::runtime_error("UKF: covariance not positive-definite");
    }

    sigmas.push_back(x_);
    for (std::size_t i = 0; i < n_; ++i) {
        Matrix col(n_, 1);
        for (std::size_t r = 0; r < n_; ++r) col(r, 0) = L(r, i);
        sigmas.push_back(add(x_, col));
    }
    for (std::size_t i = 0; i < n_; ++i) {
        Matrix col(n_, 1);
        for (std::size_t r = 0; r < n_; ++r) col(r, 0) = L(r, i);
        sigmas.push_back(sub(x_, col));
    }
}

void UnscentedKalmanFilter::predict() {
    if (!transition_) throw std::runtime_error("UKF: transition function not set");
    using namespace linalg;
    std::vector<Matrix> sigmas;
    generate_sigma_points(sigmas);

    // Push every sigma point through the transition function.
    std::vector<Matrix> propagated;
    propagated.reserve(sigmas.size());
    for (auto& sp : sigmas) {
        Matrix out(n_, 1);
        transition_(sp.data.data(), out.data.data(), transition_ctx_);
        propagated.push_back(out);
    }

    // Recombine into mean + covariance.
    Matrix mean(n_, 1);
    for (std::size_t i = 0; i < propagated.size(); ++i)
        for (std::size_t r = 0; r < n_; ++r)
            mean(r, 0) += wm_[i] * propagated[i](r, 0);

    Matrix cov(n_, n_);
    for (std::size_t i = 0; i < propagated.size(); ++i) {
        Matrix d = sub(propagated[i], mean);
        for (std::size_t r = 0; r < n_; ++r)
            for (std::size_t c = 0; c < n_; ++c)
                cov(r, c) += wc_[i] * d(r, 0) * d(c, 0);
    }
    cov = add(cov, Q_);

    x_ = mean;
    P_ = cov;
}

void UnscentedKalmanFilter::update(const Matrix& z) {
    if (!observation_) throw std::runtime_error("UKF: observation function not set");
    if (z.rows != m_ || z.cols != 1) throw std::invalid_argument("UKF.update: z shape");
    using namespace linalg;

    std::vector<Matrix> sigmas;
    generate_sigma_points(sigmas);

    std::vector<Matrix> Zsig;
    Zsig.reserve(sigmas.size());
    for (auto& sp : sigmas) {
        Matrix out(m_, 1);
        observation_(sp.data.data(), out.data.data(), observation_ctx_);
        Zsig.push_back(out);
    }

    Matrix zhat(m_, 1);
    for (std::size_t i = 0; i < Zsig.size(); ++i)
        for (std::size_t r = 0; r < m_; ++r)
            zhat(r, 0) += wm_[i] * Zsig[i](r, 0);

    Matrix S(m_, m_);
    for (std::size_t i = 0; i < Zsig.size(); ++i) {
        Matrix dz = sub(Zsig[i], zhat);
        for (std::size_t r = 0; r < m_; ++r)
            for (std::size_t c = 0; c < m_; ++c)
                S(r, c) += wc_[i] * dz(r, 0) * dz(c, 0);
    }
    S = add(S, R_);

    Matrix Cxz(n_, m_);
    for (std::size_t i = 0; i < sigmas.size(); ++i) {
        Matrix dx = sub(sigmas[i], x_);
        Matrix dz = sub(Zsig[i], zhat);
        for (std::size_t r = 0; r < n_; ++r)
            for (std::size_t c = 0; c < m_; ++c)
                Cxz(r, c) += wc_[i] * dx(r, 0) * dz(c, 0);
    }

    Matrix K = matmul(Cxz, invert_spd(S));
    x_ = add(x_, matmul(K, sub(z, zhat)));
    P_ = sub(P_, matmul(matmul(K, S), transpose(K)));
}

}  // namespace geoalpha
