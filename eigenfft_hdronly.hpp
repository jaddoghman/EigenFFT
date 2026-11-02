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
        template <typename Derived>
        auto c2c_2d(const Eigen::MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                    const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads);
    }

    template <typename Derived>
    auto fft(const Eigen::MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt, size_t axis=0,
             Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto ifft(const Eigen::MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt, size_t axis=0,
              Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto fft2(const Eigen::MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
              Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto ifft2(const Eigen::MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
               Norm norm=Norm::backward, size_t nthreads=1);
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
    using T = NumTraits<C>::Real;
    static_assert(Derived::IsVectorAtCompileTime, "Expected a vector");
    if (axis != 0)
        throw std::invalid_argument("Vector inputs cannot have axis != 0");

    const size_t m = x.size();
    const size_t N = n.value_or(m);
    if (N == 0)
        throw std::invalid_argument("Invalid FFT length");

    Vector<std::complex<T>, Dynamic> out(N);  // Output is always complex-valued
    const pocketfft::shape_t shape = {N};
    const pocketfft::shape_t axes = {axis};
    const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<T>))};
    const T fct = scale<T>(norm, N, forward);
    if constexpr ((internal::traits<Derived>::Flags & DirectAccessBit) != 0 && NumTraits<C>::IsComplex)
        if (N <= m)
        {
            const pocketfft::stride_t stride_in = {x.derived().innerStride() * static_cast<std::ptrdiff_t>(sizeof(C))};
            pocketfft::c2c(shape, stride_in, stride_out, axes, forward, x.derived().data(), out.data(), fct, nthreads);

            return out;
        }

    // Zero-pad or input isn't a concrete object
    Vector<std::complex<T>, Dynamic> in = Vector<std::complex<T>, Dynamic>::Zero(N);
    const size_t k = std::min<size_t>(N, m);
    in.head(k) = x.head(k).template cast<std::complex<T>>();
    pocketfft::c2c(shape, stride_out, stride_out, axes, forward, in.data(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::c2c_2d(const Eigen::MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                        const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads)
{
    using namespace Eigen;
    using C = Derived::Scalar;
    using T = NumTraits<C>::Real;
    if (s.has_value() && s->size() != axes.size())
        throw std::invalid_argument("Shape and axes have different lengths");
    for (const size_t a : axes)
        if (a > 1)
            throw std::out_of_range("Entries of axes must be 0 or 1");

    const size_t m = x.rows(), n = x.cols();
    size_t M = m, N = n;
    if (s.has_value())
        for (size_t i = 0; i < axes.size(); i++)
            (axes[i] == 0 ? M : N) = (*s)[i];
    if (M == 0 || N == 0)
        throw std::invalid_argument("Invalid FFT length");

    // Normalization uses only the transformed axis
    size_t nfft = 1;
    for (const size_t a : axes)
        nfft *= (a == 0 ? M : N);
    const T fct = scale<T>(norm, nfft, forward);

    Matrix<std::complex<T>, Dynamic, Dynamic> out(M, N);
    const auto sz = static_cast<std::ptrdiff_t>(sizeof(std::complex<T>));
    const pocketfft::shape_t shape = {M, N};
    const pocketfft::stride_t stride_out = {sz, static_cast<std::ptrdiff_t>(M)*sz};
    if constexpr ((internal::traits<Derived>::Flags & DirectAccessBit) != 0 && NumTraits<C>::IsComplex)
        if (M <= m && N <= n)
        {
            const pocketfft::stride_t stride_in = {x.derived().rowStride()*sz, x.derived().colStride()*sz};
            pocketfft::c2c(shape, stride_in, stride_out, axes, forward, x.derived().data(), out.data(), fct, nthreads);

            return out;
        }

    // Zero-pad or input isn't a concrete object
    Matrix<std::complex<T>, Dynamic, Dynamic> in = Matrix<std::complex<T>, Dynamic, Dynamic>::Zero(M, N);
    const size_t mm = std::min<size_t>(M, m), nn = std::min<size_t>(N, n);
    in.topLeftCorner(mm, nn) = x.topLeftCorner(mm, nn).template cast<std::complex<T>>();
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

template <typename Derived>
auto pfft::fft2(const Eigen::MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes,
                Norm norm, size_t nthreads)
{
    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2c_2d(x, opt, axes, norm, true, nthreads);
}

template <typename Derived>
auto pfft::ifft2(const Eigen::MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes,
                 Norm norm, size_t nthreads)
{
    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2c_2d(x, opt, axes, norm, false, nthreads);
}
