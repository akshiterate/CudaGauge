# `CMakeLists.txt` — the build

## 1. Purpose
The only build file. Defines the `cudagauge` static library, the
`cudagauge_cli` tool, the `sample_app` example, the three tests, and
CTest registration.

## 2. Concepts
- **CMake target**: a named thing to build (`add_library` /
  `add_executable`). `target_link_libraries(x PRIVATE cudagauge)` says
  "x uses the library".
- **`project(... LANGUAGES CXX CUDA)`**: enables nvcc for `.cu` files.
- **`CMAKE_CUDA_ARCHITECTURES 86`**: compiles GPU code for compute
  capability 8.6 = RTX 2050 (Ampere laptop). Other GPUs need this
  changed (see mistake below).
- **Generator**: NMake Makefiles was used because the CUDA toolkit's VS
  integration is absent on this machine; builds run inside the
  `vcvars64.bat` environment so `cl` is on PATH (see DECISIONS.md).

## 3–5. Walkthrough
- `cmake_minimum_required(VERSION 3.18)` + C++17/CUDA-17 standards.
- `add_library(cudagauge STATIC ...)`: 7 sources — 4 plain C++ plus 3
  CUDA (`benchmark.cu`, `workloads_gpu.cu`). `PUBLIC include` means
  consumers automatically get the headers; `PRIVATE src` keeps
  `cpu_work.h` internal.
- `add_executable(cudagauge_cli src/main.cpp)` + link.
- `add_executable(sample_app examples/sample_app.cu)` + link — deliberately
  a separate target consuming only public headers.
- `add_executable(vectorAdd_cudagauge ...)` — Level 3: NVIDIA
  cuda-samples vectorAdd behind the public adapter (upstream file itself
  is never compiled).
- Three test executables + `enable_testing()` + `add_test(...)` so
  `ctest` discovers them, plus a fourth entry running the Level 3
  adapter in `--quick` smoke mode.

## 6. Why this design
One file, no find-modules, no fetched dependencies: the whole build is
auditable in 40 lines.

## 7. Example
```
Remove-Item -Recurse -Force build   # clean reconfigure
cmd /c '"...\vcvars64.bat" >NUL && cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release'
cmd /c '"...\vcvars64.bat" >NUL && cmake --build build'
ctest --test-dir build --output-on-failure
```

## 8. Common mistakes
- Wrong GPU arch (e.g. running sm_86 binary on a sm_75 card fails).
  Reconfigure with `-DCMAKE_CUDA_ARCHITECTURES=<yours>`.
- Configuring without the vcvars environment: `cl` not found / nvcc
  cannot find a host compiler.
- Reusing a `build/` dir after switching generators (delete it first).

## 9. Dependencies
Requires CMake ≥ 3.18, MSVC 2022 BuildTools, CUDA toolkit 12.5.
No third-party libraries at all.
