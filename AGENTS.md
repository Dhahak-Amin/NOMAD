# AGENTS.md

## Cursor Cloud specific instructions

### Project Overview

NOMAD 4 (v4.5.1) is a C++ scientific computing library/application for constrained optimization of black-box functions using the MADS algorithm. It compiles into a CLI executable (`nomad`) and shared libraries. No external services, databases, or Docker containers are needed.

### Build & Run

The project uses CMake with GCC (not Clang — the default `c++` on this VM is Clang which has linker issues with `-lstdc++`). Always specify GCC explicitly:

```bash
cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DBUILD_EXAMPLES=ON -DTEST_OPENMP=ON
cmake --build build/release -j$(nproc)
cmake --install build/release
```

### Running Tests

Tests require `LD_LIBRARY_PATH` to find the shared libs and the install step to place the `nomad` binary at the expected path:

```bash
cd build/release
LD_LIBRARY_PATH=/workspace/build/release/lib:$LD_LIBRARY_PATH ctest --output-on-failure
```

Known issue: Test #3 (MultiObjBasicLib) segfaults — this is a pre-existing issue in the repository, not caused by the environment.

### Running the NOMAD executable

```bash
LD_LIBRARY_PATH=/workspace/build/release/lib:$LD_LIBRARY_PATH /workspace/build/release/bin/nomad <param_file>
```

Example: `cd examples/basic/batch/example1 && LD_LIBRARY_PATH=/workspace/build/release/lib /workspace/build/release/bin/nomad param.txt`

### Key Gotchas

- The default system `c++` compiler is Clang which fails to link (`cannot find -lstdc++`). Always pass `-DCMAKE_CXX_COMPILER=g++` to cmake.
- After building, you must run `cmake --install build/release` before CTest will find the `nomad` binary (it expects it at `build/release/bin/nomad`).
- `LD_LIBRARY_PATH` must include `build/release/lib` when running tests or the nomad executable.
