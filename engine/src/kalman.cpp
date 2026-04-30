// Linear Kalman filter implementation.
//
// The classic five-equation form, written out long-hand because the
// engine is cross-compiled to a few embedded targets where I want to
// be able to step through every operation in a debugger without
// disappearing into a templated linear algebra rabbit hole.

#include "geoalpha/state_space.hpp"

#include <cmath>
#include <stdexcept>

namespace geoalpha {

KalmanFilter::KalmanFilter(std::size_t state_dim, std::size_t obs_dim)
    : n_(state_dim), m_(obs_dim),
      F_(Matrix::identity(state_dim)),
      B_(state_dim, 0),
      H_(obs_dim, state_dim),
      Q_(Matrix::identity(state_dim)),
      R_(Matrix::identity(obs_dim)),
      x_(state_dim, 1),
      P_(Matrix::identity(state_dim)) {}

void KalmanFilter::set_transition(const Matrix& F)        { F_ = F; }
void KalmanFilter::set_control(const Matrix& B)           { B_ = B; }
void KalmanFilter::set_observation(const Matrix& H)       { H_ = H; }
void KalmanFilter::set_process_noise(const Matrix& Q)     { Q_ = Q; }
void KalmanFilter::set_measurement_noise(const Matrix& R) { R_ = R; }

void KalmanFilter::set_state(const Matrix& x, const Matrix& P) {
    if (x.rows != n_ || x.cols != 1) throw std::invalid_argument("state shape");
    if (P.rows != n_ || P.cols != n_) throw std::invalid_argument("covariance shape");
    x_ = x;
    P_ = P;
}

void KalmanFilter::predict() {
    using namespace linalg;
    x_ = matmul(F_, x_);
    P_ = add(matmul(matmul(F_, P_), transpose(F_)), Q_);
}

void KalmanFilter::predict(const Matrix& u) {
    using namespace linalg;
    if (B_.rows != n_ || B_.cols != u.rows) {
        throw std::invalid_argument("predict(u): control shape mismatch");
    }
    x_ = add(matmul(F_, x_), matmul(B_, u));
    P_ = add(matmul(matmul(F_, P_), transpose(F_)), Q_);
}

void KalmanFilter::update(const Matrix& z) {
    using namespace linalg;
    if (z.rows != m_ || z.cols != 1) throw std::invalid_argument("update: z shape");

    Matrix y = sub(z, matmul(H_, x_));                         // innovation
    Matrix S = add(matmul(matmul(H_, P_), transpose(H_)), R_); // innovation cov
    Matrix K = matmul(matmul(P_, transpose(H_)), invert_spd(S));
    x_ = add(x_, matmul(K, y));

    Matrix I = Matrix::identity(n_);
    Matrix KH = matmul(K, H_);
    P_ = matmul(sub(I, KH), P_);

    // Track ||y||_2 - useful for streaming gating / track quality.
    double s = 0.0;
    for (std::size_t i = 0; i < y.data.size(); ++i) s += y.data[i] * y.data[i];
    last_inn_ = std::sqrt(s);
}

}  // namespace geoalpha
