# geoalpha-engine (C++)

Latency-sensitive state-space estimators backing the rest of the
platform.  Linear Kalman + Unscented KF, plus a tiny linear-algebra
core (matmul / Cholesky / SPD-solve) so the library has zero deps
outside the standard library.

## build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## benchmark

```bash
./build/geoalpha_bench --steps 100000
```

Sample output on a Ryzen 7 7840U (no turbo, single thread):

```
Linear KF (4x2):   ~ 800 ns/step
UKF      (4x2):   ~ 7.5 us/step
```

## python bindings

```bash
cmake -DGEOALPHA_BUILD_PYTHON=ON ...
```

then `import geoalpha_engine_py` from Python.
