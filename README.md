# Eigen - PocketFFT

## Overview

EigenPocketFFT is a modern C++ wrapper library around PocketFFT for Eigen, targeting C++20 and later standards.
It aims to not only reproduce an FFT/DST/DCT API similar to that of NumPy and SciPy, but also to supply the user with the DST and DCT functions, which are not provided in the "unsupported" FFT module of Eigen.
Particularly, wrappers for the functions `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `rfft2`, `irfft2`, `dst`, `idst`, `dct`, `idct`, `dstn`, `idstn`, `dctn`, and `idctn` are implemented, and they support nearly identical characteristics to those of NumPy and SciPy.
All these wrappers are restricted to 1D and 2D array inputs, as Eigen does not natively support 3D arrays and beyond.
Users who are already familiar with NumPy and SciPy and have basic knowledge of Eigen can seamlessly adapt to this library.

## Usage Guide

This wrapper is a header-only library, meaning that it only requires including `eigen_pfft_hdronly.hpp`, which in turn relies on the PocketFFT header-only library `pocketfft_hdronly.h` located in the `src` directory.
Consequently, both `src` and `eigen_pfft_hdronly.hpp` must be present in the work directory to avoid compilation errors.
In what follows, the notations `np` and `sp` are adopted to represent the usual aliases of NumPy and SciPy, respectively, allowing a smooth comparison between the underlying wrapper library and Python's API.

```c++
u_hat = np.fft.fft(u)  // Python
auto u_hat = pfft::fft(u);  // EigenPocketFFT

u_hat = np.fft.fft(u, n=5, axis=0, norm="forward")  // Python
auto u_hat = pfft::fft(u, 5, 0, pfft::Norm::forward);  // EigenPocketFFT

u_hat = np.fft.rfft(u, axis=0, norm="ortho")  // Python
auto u_hat = pfft::rfft(u, std::nullopt, 0, pfft::Norm::ortho);  // EigenPocketFFT

u_hat = np.fft.irfft(u, n=7, norm="backward")  // Python
auto u_hat = pfft::irfft(u, 7, std::nullopt, pfft::Norm::backward);  // EigenPocketFFT

M_hat = np.fft.fft2(M, s=(5, 7, 3), axes=(1, 0, 1), norm="forward")  // Python
auto M_hat = pfft::fft2(u, {5, 7, 3}, {1, 0, 1}, pfft::Norm::forward);  // EigenPocketFFT

M_hat = np.fft.rfft2(M, axes=(1, 0))  // Python
auto M_hat = pfft::rfft2(M, {}, {1, 0});  // EigenPocketFFT

u_hat = np.fft.irfft2(M[1], s=(4,), axes=(0,))  // Python
auto u_hat = pfft::irfft2(M.row(1), {4}, {0});  // EigenPocketFFT

u_hat = sp.fft.dst(u, type=1, n=5, axis=0, norm="ortho", workers=1, orthogonalize=True)  // Python
auto u_hat = pfft::dst(u, 1, 5, 0, pfft::Norm::ortho, 1, true);  // EigenPocketFFT

M_hat = sp.fft.idstn(M, type=4, s=(4, 2), axes=(0, 1), norm="forward")  // Python
auto M_hat = pfft::idstn(M, 4, {4, 2}, {0, 1}, pfft::Norm::forward);  // EigenPocketFFT

M_hat = sp.fft.dctn(M, type=3, norm="backward", workers=1 orthonormalize=False)  // Python
auto M_hat = pfft::dctn(M, 3, {}, {}, pfft::Norm::backward, 1, false);  // EigenPocketFFT
```

The above code snippet provides a brief explanatory introduction to the underlying wrapper library.
The default input parameter values of EigenPocketFFT match those of NumPy and SciPy.

Here are exceptions where EigenPocketFFT slightly deviates from the definitions of NumPy and SciPy:
* Input parameters such as `n`, `axis`, `s`, `axes`, `type`, `workers` are nonnegative, as opposed to NumPy and SciPy.
* Integer-valued array inputs are not supported.
* The functions `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `rfft2`, `irfft2` have an additional input parameter `nthreads`, not shown above, which determines the number of threads for parallelization.
  Its default value is 1, but can be set to 0 in order to activate the maximum number of threads (`os.cpu_count()` is the equivalent Python command).
  If two threads are required to compute a 2D FFT with default inputs, the syntax reads 
  ```c++
  auto hat = pfft::fft2(M, {}, {0, 1}, pfft::Norm::backward, 2);
  ```

The user may refer to the documentation of NumPy and SciPy for more details about the wrapped functions, while taking into account the above few exceptions.

## Requirements

* Eigen
* C++ 20 or beyond
* PocketFFT header-only file `pocketfft_hdronly.h` provided in `src`

## Third-party dependencies

This project is based on the following licensed third-party libraries:

* **PocketFFT** - BSD 3-Clause License.
* **Eigen** - Primarily licensed under the Mozilla Public License 2.0 (MPL-2.0), with EIGEN_MPL2_ONLY enabled for the underlying project.

## Purpose

This repository was developed for scientists who appreciate the readability of Python's API, but require a higher-performance environment for their projects, while also being familiar with the C++ Eigen library.