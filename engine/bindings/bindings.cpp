// pybind11 bindings: expose the C++ engine to the Python quant module.
//
// Buffers cross the Python <-> C++ boundary as numpy arrays without
// copy where possible.  Build via:
//
//   cmake -DGEOALPHA_BUILD_PYTHON=ON ...

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "geoalpha/state_space.hpp"

namespace py = pybind11;
using namespace geoalpha;

namespace {

Matrix from_numpy(py::array_t<double, py::array::c_style | py::array::forcecast> a) {
    if (a.ndim() == 1) {
        Matrix M(static_cast<std::size_t>(a.shape(0)), 1);
        std::memcpy(M.data.data(), a.data(), a.nbytes());
        return M;
    }
    if (a.ndim() != 2) throw std::invalid_argument("expected 1- or 2-D array");
    Matrix M(static_cast<std::size_t>(a.shape(0)), static_cast<std::size_t>(a.shape(1)));
    std::memcpy(M.data.data(), a.data(), a.nbytes());
    return M;
}

py::array_t<double> to_numpy(const Matrix& M) {
    py::array_t<double> arr({M.rows, M.cols});
    std::memcpy(arr.mutable_data(), M.data.data(), M.data.size() * sizeof(double));
    return arr;
}

}  // namespace

PYBIND11_MODULE(geoalpha_engine_py, m) {
    m.doc() = "C++ state-space estimators for geo-alpha";

    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<std::size_t, std::size_t>())
        .def("set_transition",         [](KalmanFilter& kf, py::array_t<double> F)        { kf.set_transition(from_numpy(F)); })
        .def("set_observation",        [](KalmanFilter& kf, py::array_t<double> H)        { kf.set_observation(from_numpy(H)); })
        .def("set_process_noise",      [](KalmanFilter& kf, py::array_t<double> Q)        { kf.set_process_noise(from_numpy(Q)); })
        .def("set_measurement_noise",  [](KalmanFilter& kf, py::array_t<double> R)        { kf.set_measurement_noise(from_numpy(R)); })
        .def("set_state",              [](KalmanFilter& kf, py::array_t<double> x, py::array_t<double> P) { kf.set_state(from_numpy(x), from_numpy(P)); })
        .def("predict",                [](KalmanFilter& kf)                                { kf.predict(); })
        .def("update",                 [](KalmanFilter& kf, py::array_t<double> z)        { kf.update(from_numpy(z)); })
        .def("state",                  [](KalmanFilter& kf)                                { return to_numpy(kf.state()); })
        .def("covariance",             [](KalmanFilter& kf)                                { return to_numpy(kf.covariance()); });
}
