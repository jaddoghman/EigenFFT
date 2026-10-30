#pragma once

#include <Eigen/Dense>
#include "src/pocketfft_hdronly.h"

// ==========================================Declarations==========================================
namespace pfft
{
    enum class Norm { backward, ortho, forward };

    namespace core
    {
        template <typename T>
        T scale(Norm norm, size_t N, bool forward_dir);
        template <typename Derived>
        auto c2c_1d(const Eigen::MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, bool forward,
                    size_t nthreads);
    }

    template <typename Derived>
    auto fft(const Eigen::MatrixBase<Derived> &x, std::optional<size_t> n = std::nullopt, size_t axis=0,
             Norm norm = Norm::backward, size_t nthreads = 1);
    template <typename Derived>
    auto ifft(const Eigen::MatrixBase<Derived> &x, std::optional<size_t> n = std::nullopt, size_t axis=0,
              Norm norm = Norm::backward, size_t nthreads = 1);
}


// ==========================================Definitions==========================================
template <typename T>
T pfft::core::scale(Norm norm, size_t N, bool forward_dir)
{
    if (norm == Norm::ortho)
        return static_cast<T>(1) / std::sqrt(static_cast<T>(N));
    const bool is_scaled = (norm == Norm::forward) == forward_dir;
    return is_scaled ? static_cast<T>(1) / static_cast<T>(N) : static_cast<T>(1);
}

template <typename Derived>
auto pfft::core::c2c_1d(const Eigen::MatrixBase<Derived> &x, std::optional<size_t> n, size_t axis, Norm norm,
                        bool forward, size_t nthreads)
{
    using namespace Eigen;
    using C = Derived::Scalar;
    using T = C::value_type;
    static_assert(NumTraits<C>::IsComplex, "c2c requires a complex input");
    static_assert(Derived::IsVectorAtCompileTime, "Expected a vector");
    if (axis != 0)
        throw std::invalid_argument("Vector inputs cannot have axis != 0");

    const size_t sz = x.size();
    const size_t N = n.value_or(sz);
    if (N == 0)
        throw std::invalid_argument("Invalid FFT length");

    Vector<C, Dynamic> out(N);
    pocketfft::shape_t shape = {N};
    pocketfft::shape_t axes = {axis};
    pocketfft::stride_t stride_out = {sizeof(C)};
    const T fct = scale<T>(norm, N, forward);
    if constexpr ((internal::traits<Derived>::Flags & DirectAccessBit) != 0)  // x is concrete variable
        if (N <= sz)
        {
            pocketfft::stride_t stride_in = {x.innerStride() * static_cast<std::ptrdiff_t>(sizeof(C))};
            pocketfft::c2c(shape, stride_in, stride_out, axes, forward, x.derived().data(), out.data(), fct, nthreads);

            return out;
        }
    // Zero-pad or input isn't a concrete object
    const size_t m = std::min<size_t>(N, sz);
    Vector<C, Dynamic> in = Vector<C, Dynamic>::Zero(N);
    in.head(m) = x.head(m);
    pocketfft::c2c(shape, stride_out, stride_out, axes, forward, in.data(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::fft(const Eigen::MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    return core::c2c_1d(x, n, axis, norm, true, nthreads);
}

template <typename Derived>
auto pfft::ifft(const Eigen::MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    return core::c2c_1d(x, n, axis, norm, false, nthreads);
}
