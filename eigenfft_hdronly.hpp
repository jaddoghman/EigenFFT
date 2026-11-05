#pragma once

#include <Eigen/Dense>
#include "src/pocketfft_hdronly.h"

// ==========================================Declarations==========================================
namespace pfft
{
    using namespace Eigen;
    enum class Norm { backward, ortho, forward };

    namespace core
    {
        using namespace Eigen;
        template <typename Derived, typename T>
        struct InputView1D;  // Manage 1D arrays size and input size of FFT
        template <typename Derived, typename T>
        struct InputView2D;  // Manage 2D arrays size and input size of FFT
        template <typename T>
        T scale(Norm norm, size_t N, bool forward_dir);
        inline bool hasDuplicates(const pocketfft::shape_t& vec);
        template <typename T>
        auto resizeAxis(const Matrix<T, Dynamic, Dynamic>& mat, size_t axis, size_t L);
        template <typename Derived>
        auto c2c_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, bool forward,
                    size_t nthreads);
        template <typename Derived>
        auto c2c_dplct(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                       const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads);
        template <typename Derived>
        auto c2c_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                    const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads);
        template <typename Derived>
        auto irfft_vec(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads);
        template <typename Derived>
        auto irfft_mat(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads);
    }

    template <typename Derived>
    auto fft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt, size_t axis=0,
             Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto ifft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt, size_t axis=0,
              Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto fft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
              Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto ifft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
               Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto rfft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt, size_t axis=0,
              Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto irfft(const MatrixBase<Derived>& x, std::optional<size_t>n=std::nullopt,
               size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t nthreads=1);
}


// ==========================================Definitions==========================================
template <typename Derived, typename OutputT>
struct pfft::core::InputView1D
{
private:
    using InputT = Derived::Scalar;  // Input type
    using RType = std::conditional_t<std::is_same_v<InputT, OutputT>, InputT, OutputT>;  // Resulting type
    std::ptrdiff_t mStride;
    const RType* mPtr;
    Vector<OutputT, Dynamic> mBuf;  // If input isn't a concrete object or has ≠ type
public:
    InputView1D() : mStride(0), mPtr(nullptr), mBuf{} { }
    InputView1D(const InputView1D&) = delete;
    InputView1D(InputView1D&&) = delete;
    ~InputView1D() = default;
    explicit InputView1D(const MatrixBase<Derived>& x, size_t len)
    {
        const size_t m = x.size();
        if constexpr ((internal::traits<Derived>::Flags & DirectAccessBit) != 0 && std::is_same_v<InputT, OutputT>)
            if (len <= m)
            {
                this->mStride = static_cast<std::ptrdiff_t>(x.derived().innerStride() * sizeof(InputT));
                this->mPtr = x.derived().data();
                return;
            }

        // Zero-pad or input x isn't a concrete object or input x has different type
        this->mBuf = Vector<OutputT, Dynamic>::Zero(len);
        const size_t mm = std::min<size_t>(len, m);
        this->mBuf.head(mm) = x.head(mm).template cast<OutputT>();
        this->mPtr = mBuf.data();
        this->mStride = static_cast<std::ptrdiff_t>(sizeof(OutputT));
    }
    [[nodiscard]] std::ptrdiff_t stride() const { return mStride; }
    [[nodiscard]] const RType* ptr() const { return mPtr; }
};

template <typename Derived, typename OutputT>
struct pfft::core::InputView2D
{
private:
    using InputT = Derived::Scalar;  // Input type
    using RType = std::conditional_t<std::is_same_v<InputT, OutputT>, InputT, OutputT>;  // Resulting type
    pocketfft::stride_t mStride;
    const RType* mPtr;
    Matrix<OutputT, Dynamic, Dynamic> mBuf;  // If input isn't a concrete object or has ≠ type
public:
    InputView2D() = default;
    InputView2D(const InputView2D&) = delete;
    InputView2D(InputView2D&&) = delete;
    ~InputView2D() = default;
    explicit InputView2D(const MatrixBase<Derived>& x, size_t r, size_t c)
    {
        const size_t m = x.rows(), n = x.cols();
        if constexpr ((internal::traits<Derived>::Flags & DirectAccessBit) != 0 && std::is_same_v<InputT, OutputT>)
            if (r <= m && c <= n)
            {
                constexpr auto sz = static_cast<std::ptrdiff_t>(sizeof(InputT));
                this->mStride = {x.derived().rowStride()*sz, x.derived().colStride()*sz};
                this->mPtr = x.derived().data();
                return;
            }

        // Zero-pad or input x isn't a concrete object or input x has ≠ type
        this->mBuf = Matrix<OutputT, Dynamic, Dynamic>::Zero(r, c);
        const size_t rr = std::min<size_t>(r, m), cc = std::min<size_t>(c, n);
        this->mBuf.topLeftCorner(rr, cc) = x.topLeftCorner(rr, cc).template cast<OutputT>();
        constexpr auto sz = static_cast<std::ptrdiff_t>(sizeof(OutputT));
        this->mStride = {mBuf.rowStride()*sz, mBuf.colStride()*sz};
        this->mPtr = mBuf.data();
    }
    [[nodiscard]] const pocketfft::stride_t& stride() const { return mStride; }
    [[nodiscard]] const RType* ptr() const { return mPtr; }
};

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
auto pfft::core::resizeAxis(const Matrix<T, Dynamic, Dynamic>& mat, size_t axis, size_t L)
{
    const size_t m = mat.rows(), n = mat.cols();
    const size_t r = axis == 0 ? L : m;
    const size_t c = axis == 1 ? L : n;
    Matrix<T, Dynamic, Dynamic> out = Matrix<T, Dynamic, Dynamic>::Zero(r, c);
    const size_t rr = std::min<size_t>(r, m), cc = std::min<size_t>(c, n);
    out.topLeftCorner(rr, cc) = mat.topLeftCorner(rr, cc);

    return out;
}

template <typename Derived>
auto pfft::core::c2c_1d(const MatrixBase<Derived> &x, std::optional<size_t> n, size_t axis, Norm norm, bool forward,
                        size_t nthreads)
{
    using C = Derived::Scalar;
    using R = NumTraits<C>::Real;
    static_assert(Derived::IsVectorAtCompileTime, "Expected a 1D array");
    if (axis != 0)
        throw std::invalid_argument("1D array inputs are not allowed to have axis != 0");

    const size_t m = x.size();
    const size_t M = n.value_or(m);
    if (M == 0)
        throw std::invalid_argument("Invalid FFT length");

    Vector<std::complex<R>, Dynamic> out(M);  // Output is always complex-valued
    const pocketfft::shape_t shape = {M};
    const pocketfft::shape_t axes = {axis};
    const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<R>))};
    const R fct = scale<R>(norm, M, forward);

    InputView1D<Derived, std::complex<R>> in(x, M);
    pocketfft::c2c(shape, {in.stride()}, stride_out, axes, forward, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::c2c_dplct(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                           const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads)
{
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
auto pfft::core::c2c_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                         const pocketfft::shape_t& axes, Norm norm, bool forward, size_t nthreads)
{
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
    InputView2D<Derived, std::complex<T>> in(x, M, N);
    const pocketfft::shape_t shape = {M, N};
    constexpr auto sz = static_cast<std::ptrdiff_t>(sizeof(std::complex<T>));
    const pocketfft::stride_t stride_out = {sz, static_cast<std::ptrdiff_t>(M)*sz};
    pocketfft::c2c(shape, in.stride(), stride_out, axes, forward, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::irfft_vec(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm,
                           size_t nthreads)
{
    using S = Derived::Scalar;  // Can be real or complex
    using R = NumTraits<S>::Real;
    if (Derived::IsVectorAtCompileTime && axis != 0)
        throw std::out_of_range("1D array inputs are not allowed to have axis != 0");

    const size_t m = x.size();
    const size_t M = n.value_or(m < 2 ? 0 : 2*(m-1));  // Output length
    if (M == 0)
        throw std::invalid_argument("Invalid FFT length");
    const size_t N = M/2 + 1;  // Complex samples needed
    const R fct = core::scale<R>(norm, M, false);

    InputView1D<Derived, std::complex<R>> in(x, N);
    Vector<R, Dynamic> out(M);  // Real-valued output
    const pocketfft::shape_t shape = {M};
    const pocketfft::shape_t axes = {axis};
    const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(R))};
    pocketfft::c2r(shape, {in.stride()}, stride_out, axes, false, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::irfft_mat(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm,
                           size_t nthreads)
{
    using S = Derived::Scalar;  // Can be real of complex
    using R = NumTraits<S>::Real;
    if (axis > 1)
        throw std::out_of_range("Entries of axes must be 0 or 1");

    const size_t r = x.rows(), c = x.cols();
    const size_t M = axis == 0 ? n.value_or(r < 2 ? 0 : 2*(r-1)) : r;
    const size_t N = axis == 1 ? n.value_or(c < 2 ? 0 : 2*(c-1)) : c;
    Matrix<R, Dynamic, Dynamic> out(M, N);

    if (axis == 0)
        for (size_t j = 0; j < N; j++)
            out.col(j) = irfft_vec(x.col(j), n, 0, norm, nthreads);
    else
        for (size_t i = 0; i < M; i++)
            out.row(i) = irfft_vec(x.row(i), n, 0, norm, nthreads);


    return out;
}

template <typename Derived>
auto pfft::fft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    return core::c2c_1d(x, n, axis, norm, true, nthreads);
}

template <typename Derived>
auto pfft::ifft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    return core::c2c_1d(x, n, axis, norm, false, nthreads);
}

template <typename Derived>
auto pfft::fft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes, Norm norm,
                size_t nthreads)
{
    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2c_2d(x, opt, axes, norm, true, nthreads);
}

template <typename Derived>
auto pfft::ifft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes, Norm norm,
                 size_t nthreads)
{
    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2c_2d(x, opt, axes, norm, false, nthreads);
}

template <typename Derived>
auto pfft::rfft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    using R = Derived::Scalar;
    static_assert(!NumTraits<R>::IsComplex, "rfft input must be real");
    static_assert(Derived::IsVectorAtCompileTime, "Expected a 1D array");
    if (axis != 0)
        throw std::out_of_range("1D array inputs are not allowed to have axis != 0");

    const size_t m = x.size();
    const size_t N = n.value_or(m);  // Real transform length
    if (N == 0)
        throw std::invalid_argument("Invalid FFT length");
    const size_t M = N/2 + 1;  // Output length

    core::InputView1D<Derived, R> in(x, N);
    Vector<std::complex<R>, Dynamic> out(M);  // Complex-valued output
    const pocketfft::shape_t shape = {N};
    const pocketfft::shape_t axes = {axis};
    const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<R>))};
    const R fct = core::scale<R>(norm, N, true);  // Normalization factor
    pocketfft::r2c(shape, {in.stride()}, stride_out, axes, true, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::irfft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::irfft_vec(x, n, axis, norm, nthreads);
    else if constexpr (Derived::NumDimensions == 2)
        return core::irfft_mat(x, n, axis, norm, nthreads);
    else
        throw std::invalid_argument("Input must be a 1D or 2D array");
}
