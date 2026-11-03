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
        inline bool hasDuplicates(const pocketfft::shape_t& vec);
        template <typename T>
        auto resizeAxis(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat, size_t axis, size_t L);

        template <typename Derived>
        auto c2c_1d(const Eigen::MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, bool forward,
                    size_t nthreads);
        template <typename Derived>
        auto c2c_dplct(const Eigen::MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                       const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads);
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

bool pfft::core::hasDuplicates(const pocketfft::shape_t& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
        for (size_t j = i+1; j < vec.size(); j++)
            if (vec[i] == vec[j])
                return true;
    return false;
}

template <typename T>
auto pfft::core::resizeAxis(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>& mat, size_t axis, size_t L)
{
    using namespace Eigen;
    const size_t m = mat.rows(), n = mat.cols();
    const size_t r = axis == 0 ? L : m;
    const size_t c = axis == 1 ? L : n;
    Matrix<T, Dynamic, Dynamic> out = Matrix<T, Dynamic, Dynamic>::Zero(r, c);
    const size_t rr = std::min<size_t>(r, m), cc = std::min<size_t>(c, n);
    out.topLeftCorner(rr, cc) = mat.topLeftCorner(rr, cc);

    return out;
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
auto pfft::core::c2c_dplct(const Eigen::MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                           const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads)
{
    using namespace Eigen;
    using T = NumTraits<typename Derived::Scalar>::Real;
    constexpr auto cplx_sz = static_cast<std::ptrdiff_t>(sizeof(std::complex<T>));
    Matrix<std::complex<T>, Dynamic, Dynamic> out = x.template cast<std::complex<T>>();
    for (size_t i = axes.size(); i-- > 0; )
    {
        const size_t cur_ax = axes[i];  // Current axis
        const size_t cur_len = cur_ax == 0 ? out.rows() : out.cols();
        const size_t new_len = s.has_value() ? (*s)[i] : cur_len;

        if (new_len == 0 || cur_len == 0)
            throw std::invalid_argument("Invalid FFT length");

        if (cur_len != new_len)
            out = resizeAxis(out, cur_ax, new_len);  // Zero-pad or crop

        const pocketfft::shape_t shape = {static_cast<size_t>(out.rows()), static_cast<size_t>(out.cols())};
        const pocketfft::stride_t stride = {cplx_sz, static_cast<std::ptrdiff_t>(out.rows())*cplx_sz};
        const T fct = scale<T>(norm, new_len, forward);  // Normalization factor
        pocketfft::c2c(shape, stride, stride, {cur_ax}, forward, out.data(), out.data(), fct, nthreads);
    }
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
        throw std::invalid_argument("Shape and axes must have same lengths");
    const size_t axes_max = *std::ranges::max_element(axes.begin(), axes.end());
    if (axes_max > 1)
        throw std::out_of_range("Entries of axes must be 0 or 1");
    if ((x.rows() == 1 || x.cols() == 1) && axes_max != 0)
        throw std::out_of_range("Column/Row matrices do not support non-zero axes entries");

    if (hasDuplicates(axes))
        return core::c2c_dplct(x, s, axes, norm, forward, nthreads);

    // The remaining part of the code assumes no duplication in axes
    const size_t m = x.rows(), n = x.cols();
    size_t M = m, N = n;  // Output shape
    if (s.has_value())
        for (size_t i = 0; i < axes.size(); i++)
            (axes[i] == 0 ? M : N) = (*s)[i];
    if (M == 0 || N == 0)
        throw std::invalid_argument("Invalid FFT length");

    size_t nfft = 1;  // To normalize along the transformed axis only
    for (const size_t a : axes)
        nfft *= (a == 0 ? M : N);
    const T fct = scale<T>(norm, nfft, forward);  // Normalization factor

    Matrix<std::complex<T>, Dynamic, Dynamic> out(M, N);
    const pocketfft::shape_t shape = {M, N};
    const auto cplx_sz = static_cast<std::ptrdiff_t>(sizeof(std::complex<T>));
    const pocketfft::stride_t stride_out = {cplx_sz, static_cast<std::ptrdiff_t>(M)*cplx_sz};

    if constexpr ((internal::traits<Derived>::Flags & DirectAccessBit) != 0 && NumTraits<C>::IsComplex)
        if (M <= m && N <= n)
        {
            const pocketfft::stride_t stride_in = {x.derived().rowStride()*cplx_sz, x.derived().colStride()*cplx_sz};
            pocketfft::c2c(shape, stride_in, stride_out, axes, forward, x.derived().data(), out.data(), fct, nthreads);

            return out;
        }

    // Zero-pad or input isn't a concrete object or input isn't complex-valued
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
