#pragma once

#include <Eigen/Dense>
#include "src/pocketfft_hdronly.h"

// ==========================================Declarations==========================================
namespace pfft
{
    using namespace Eigen;
    enum class Norm { backward, ortho, forward };  // Normalization types

    namespace core
    {
        using namespace Eigen;
        template <typename Derived, typename OutputT>
        struct InputView1D;  // Manage 1D arrays size and input size of FFT
        template <typename Derived, typename OutputT>
        struct InputView2D;  // Manage 2D arrays size and input size of FFT
        template <typename T>
        T scale(Norm norm, size_t N, bool forward_dir);
        inline bool hasDuplicates(const pocketfft::shape_t& vec);
        inline void assertConditions(size_t r, size_t c, const std::optional<pocketfft::shape_t>& s,
                                     const pocketfft::shape_t& axes);
        template <bool forward, typename Derived>
        auto c2c_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads);
        template <bool forward, typename Derived>
        auto c2c_1d_extrude(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm,
                            size_t nthreads);
        template <typename Derived>
        auto r2c_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads);
        template <typename Derived>
        auto c2r_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads);
        template <bool forward, typename Derived>
        auto c2c_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s, pocketfft::shape_t axes,
                    Norm norm, size_t nthreads);
        template <typename Derived>
        auto r2c_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s, pocketfft::shape_t axes,
                    Norm norm, size_t nthreads);
        template <typename Derived>
        auto c2r_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s, pocketfft::shape_t axes,
                    Norm norm, size_t nthreads);
        template <bool forward, typename Derived>
        auto dst_1d(const MatrixBase<Derived>& x, size_t type, std::optional<size_t> n, size_t axis, Norm norm,
                    size_t workers, std::optional<bool> orthogonalize);
        template <bool forward, typename Derived>
        auto dst_1d_extrude(const MatrixBase<Derived>& x, size_t type, std::optional<size_t> n, size_t axis, Norm norm,
                            size_t workers, std::optional<bool> orthogonalize);
    }

    template <typename Derived>
    auto fft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt,
             size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto ifft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt,
              size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto rfft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt,
              size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto irfft(const MatrixBase<Derived>& x, std::optional<size_t> n=std::nullopt,
               size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto fft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
              Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto ifft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
               Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto rfft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
               Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto irfft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s={}, const pocketfft::shape_t& axes={0,1},
                Norm norm=Norm::backward, size_t nthreads=1);
    template <typename Derived>
    auto dst(const MatrixBase<Derived>& x, size_t type=2, std::optional<size_t> n=std::nullopt,
             size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t workers=1,
             std::optional<bool> orthogonalize=std::nullopt);
    template <typename Derived>
    auto idst(const MatrixBase<Derived>& x, size_t type=2, std::optional<size_t> n=std::nullopt,
              size_t axis=Derived::IsVectorAtCompileTime ? 0:1, Norm norm=Norm::backward, size_t workers=1,
              std::optional<bool> orthogonalize=std::nullopt);
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

inline bool pfft::core::hasDuplicates(const pocketfft::shape_t& vec)
{
    for (size_t i = 0; i < vec.size(); i++)
        for (size_t j = i+1; j < vec.size(); j++)
            if (vec[i] == vec[j])
                return true;
    return false;
}

inline void pfft::core::assertConditions(size_t r, size_t c, const std::optional<pocketfft::shape_t>& s,
                                         const pocketfft::shape_t& axes)
{
    if (s.has_value() && s->size() != axes.size())
        throw std::invalid_argument("Shape and axes must have same lengths");
    const size_t axes_max = *std::ranges::max_element(axes.begin(), axes.end());
    if (axes_max > 1)
        throw std::out_of_range("Entries of axes must be 0 or 1");
    if ((r == 1 || c == 1) && axes_max != 0)
        throw std::out_of_range("1D array inputs do not support non-zero axes entries");
}

template <bool forward, typename Derived>
auto pfft::core::c2c_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
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

    InputView1D<Derived, std::complex<R>> in(x, M);  // Zero-pad or crop
    pocketfft::c2c(shape, {in.stride()}, stride_out, axes, forward, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <bool forward, typename Derived>
auto pfft::core::c2c_1d_extrude(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm,
                                size_t nthreads)
{
    using R = NumTraits<typename Derived::Scalar>::Real;
    if (axis > 1)
        throw std::out_of_range("Entries of axis must be 0 or 1");

    const size_t r = x.rows(), c = x.cols();
    const size_t M = axis == 0 ? n.value_or(r) : r;
    const size_t N = axis == 1 ? n.value_or(c) : c;
    if (M == 0 || N == 0)
        throw std::invalid_argument("Invalid FFT length");

    Matrix<std::complex<R>, Dynamic, Dynamic> out(M, N);
    if (axis == 0)
        for (size_t j = 0; j < N; j++)
            out.col(j) = core::c2c_1d<forward>(x.col(j), n, 0, norm, nthreads);
    else
        for (size_t i = 0; i < M; i++)
            out.row(i) = core::c2c_1d<forward>(x.row(i), n, 0, norm, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::r2c_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    using R = Derived::Scalar;
    static_assert(!NumTraits<R>::IsComplex, "rfft inputs must be real");
    if (axis != 0)
        throw std::out_of_range("1D array inputs are not allowed to have axis != 0");

    const size_t m = x.size();
    const size_t M = n.value_or(m);  // Real transform length
    if (M == 0)
        throw std::invalid_argument("Invalid FFT length");
    const size_t N = M/2 + 1;  // Output length

    InputView1D<Derived, R> in(x, M);  // Zero-pad or crop
    Vector<std::complex<R>, Dynamic> out(N);  // Complex-valued output
    const pocketfft::shape_t shape = {M};
    const pocketfft::shape_t axes = {axis};
    const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(std::complex<R>))};
    const R fct = core::scale<R>(norm, M, true);  // Normalization factor
    pocketfft::r2c(shape, {in.stride()}, stride_out, axes, true, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::c2r_1d(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    using S = Derived::Scalar;  // Could be real or complex
    using R = NumTraits<S>::Real;
    static_assert(Derived::IsVectorAtCompileTime, "Expected a 1D array");
    if (axis != 0)
        throw std::out_of_range("1D array inputs are not allowed to have axis != 0");

    const size_t m = x.size();
    const size_t M = n.value_or(m < 2 ? 0 : 2*(m-1));  // Output length
    if (M == 0)
        throw std::invalid_argument("Invalid FFT length");
    const size_t N = M/2 + 1;  // Complex samples needed
    const R fct = core::scale<R>(norm, M, false);

    InputView1D<Derived, std::complex<R>> in(x, N);  // Zero-pad or crop
    Vector<R, Dynamic> out(M);  // Real-valued output
    const pocketfft::shape_t shape = {M};
    const pocketfft::shape_t axes = {axis};
    const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(R))};
    pocketfft::c2r(shape, {in.stride()}, stride_out, axes, false, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <bool forward, typename Derived>
auto pfft::core::c2c_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                        pocketfft::shape_t axes, Norm norm, size_t nthreads)
{
    using R = NumTraits<typename Derived::Scalar>::Real;
    assertConditions(x.rows(), x.cols(), s, axes);

    if (hasDuplicates(axes))
    {
        Matrix<std::complex<R>, Dynamic, Dynamic> out = x.template cast<std::complex<R>>();
        for (size_t k = axes.size(); k-- > 0; )
        {
            const size_t cur_axis = axes[k];
            const size_t cur_len = cur_axis == 0 ? out.rows() : out.cols();
            const size_t new_len = s.has_value() ? (*s)[k] : cur_len;

            if (new_len == 0 || cur_len == 0)
                throw std::invalid_argument("Invalid FFT length");

            if constexpr (forward)
                out = fft(out, new_len, cur_axis, norm, nthreads);
            else
                out = ifft(out, new_len, cur_axis, norm, nthreads);
        }
        return out;
    }

    // The remaining case handles unduplicated axes
    const size_t m = x.rows(), n = x.cols();
    if (m == 1 && n != 1) // x is row matrix
        std::ranges::fill(axes.begin(), axes.end(), 1);
    size_t M = m, N = n;  // Output shape if s is empty
    if (s.has_value())
        for (size_t i = 0; i < axes.size(); i++)
            (axes[i] == 0 ? M : N) = (*s)[i];
    if (M == 0 || N == 0)
        throw std::invalid_argument("Invalid FFT length");
    size_t nfft = 1;  // Normalize along the transformed axes only
    for (const size_t a : axes)
        nfft *= (a == 0 ? M : N);
    const R fct = scale<R>(norm, nfft, forward);  // Normalization factor

    Matrix<std::complex<R>, Dynamic, Dynamic> out(M, N);
    InputView2D<Derived, std::complex<R>> in(x, M, N);  // Zero-pad or crop
    const pocketfft::shape_t shape = {M, N};
    constexpr auto csz = static_cast<std::ptrdiff_t>(sizeof(std::complex<R>));
    const pocketfft::stride_t stride_out = {csz, static_cast<std::ptrdiff_t>(M)*csz};
    pocketfft::c2c(shape, in.stride(), stride_out, axes, forward, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::r2c_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                        pocketfft::shape_t axes, Norm norm, size_t nthreads)
{
    using R = Derived::Scalar;
    static_assert(!NumTraits<R>::IsComplex, "rfft2 input must be real");
    assertConditions(x.rows(), x.cols(), s, axes);

    if (hasDuplicates(axes))
    {
        const size_t last_axis = axes.back();
        const size_t len = s.has_value() ? s->back() : (last_axis == 0 ? x.rows() : x.cols());
        Matrix<std::complex<R>, Dynamic, Dynamic> out = rfft(x, len, last_axis, norm, nthreads);

        for (size_t k = axes.size()-1; k-- > 0; )
        {
            const size_t cur_axis = axes[k];
            const size_t cur_len = cur_axis == 0 ? out.rows() : out.cols();
            const size_t new_len = s.has_value() ? (*s)[k] : cur_len;

            if (new_len == 0 || cur_len == 0)
                throw std::invalid_argument("Invalid FFT length");

            out = fft(out, new_len, cur_axis, norm, nthreads);
        }
        return out;
    }

    // The remaining case handles unduplicated axes
    size_t m = x.rows(), n = x.cols();
    if (m == 1 && n != 1)  // x is row matrix
        std::ranges::fill(axes.begin(), axes.end(), 1);
    for (size_t k = 0; k < axes.size(); k++)
    {
        if (!s.has_value())
            break;
        (axes[k] == 0 ? m : n) = (*s)[k];
    }
    if (m == 0 || n == 0)
        throw std::invalid_argument("Invalid FFT length");
    const size_t M = axes.back() == 0 ? m/2+1 : m;
    const size_t N = axes.back() == 1 ? n/2+1 : n;
    size_t nfft = 1;  // Normalize along the transformed axes only
    for (const size_t a : axes)
        nfft *= (a == 0 ? m : n);
    const R fct = scale<R>(norm, nfft, true);

    Matrix<std::complex<R>, Dynamic, Dynamic> out(M, N);
    InputView2D<Derived, R> in(x, m, n);  // Zero-pad or crop
    constexpr auto csz = static_cast<std::ptrdiff_t>(sizeof(std::complex<R>));
    const pocketfft::shape_t shape = {m, n};
    const pocketfft::stride_t stride_out = {csz, static_cast<std::ptrdiff_t>(M)*csz};
    pocketfft::r2c(shape, in.stride(), stride_out, axes, true, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <typename Derived>
auto pfft::core::c2r_2d(const MatrixBase<Derived>& x, const std::optional<pocketfft::shape_t>& s,
                        pocketfft::shape_t axes, Norm norm, size_t nthreads)
{
    using R = NumTraits<typename Derived::Scalar>::Real;
    assertConditions(x.rows(), x.cols(), s, axes);

    if (hasDuplicates(axes))
    {
        size_t len = s.has_value() ? s->front() : (axes.front() == 0 ? x.rows() : x.cols());
        Matrix<std::complex<R>, Dynamic, Dynamic> in = ifft(x, len, axes.front(), norm, nthreads);

        for (size_t k = 1; k < axes.size()-1; k++)
        {
            const size_t cur_len = axes[k] == 0 ? in.rows() : in.cols();
            const size_t new_len = s.has_value() ? (*s)[k] : cur_len;

            if (new_len == 0 || cur_len == 0)
                throw std::invalid_argument("Invalid FFT length");

            in = ifft(in, new_len, axes[k], norm, nthreads);
        }
        len = s.has_value() ? s->back() : (axes.back() == 0 ? 2*(x.rows()-1) : 2*(x.cols()-1));

        return irfft(in, len, axes.back(), norm, nthreads);
    }

    // The remaining case handles unduplicated axes
    const size_t m = x.rows(), n = x.cols();
    if (m == 1 && n != 1)  // x is row matrix
        std::ranges::fill(axes.begin(), axes.end(), 1);
    size_t M = axes.back() == 0 ? 2*(m-1) : m;
    size_t N = axes.back() == 1 ? 2*(n-1) : n;
    if (s.has_value())
        for (size_t k = 0; k < axes.size(); k++)
            (axes[k] == 0 ? M : N) = (*s)[k];
    if (M == 0 || N == 0)
        throw std::invalid_argument("Invalid FFT length");
    size_t nfft = 1;  // Normalize along transformed axes only
    for (const size_t a : axes)
        nfft *= (a == 0 ? M : N);
    const R fct = scale<R>(norm, nfft, false);

    Matrix<R, Dynamic, Dynamic> out(M, N);
    const size_t r = axes.back() == 0 ? M/2+1 : M;
    const size_t c = axes.back() == 1 ? N/2+1 : N;
    InputView2D<Derived, std::complex<R>> in(x, r, c);  // Zero-pad or crop
    constexpr auto rsz = static_cast<std::ptrdiff_t>(sizeof(R));
    const pocketfft::shape_t shape = {M, N};
    const pocketfft::stride_t stride_out = {rsz, static_cast<std::ptrdiff_t>(M)*rsz};
    pocketfft::c2r(shape, in.stride(), stride_out, axes, false, in.ptr(), out.data(), fct, nthreads);

    return out;
}

template <bool forward, typename Derived>
auto pfft::core::dst_1d(const MatrixBase<Derived>& x, size_t type, std::optional<size_t> n, size_t axis, Norm norm,
                        size_t workers, std::optional<bool> orthogonalize)
{
    using S = Derived::Scalar;
    using R = NumTraits<S>::Real;
    static_assert(Derived::IsVectorAtCompileTime, "Expected a 1D array");
    if (type == 0 || type > 4)
        throw std::invalid_argument("Invalid DST type");
    if (axis != 0)
        throw std::out_of_range("1D array inputs are not allowed to have axis != 0");

    if constexpr (NumTraits<S>::IsComplex)  // dst takes in only real inputs
    {
        Matrix<R, Dynamic, Dynamic> re = dst_1d<forward>(x.real(), type, n, axis, norm, workers, orthogonalize);
        Matrix<R, Dynamic, Dynamic> im = dst_1d<forward>(x.imag(), type, n, axis, norm, workers, orthogonalize);
        Vector<S, Dynamic> out(re.size());
        out.real() = re;
        out.imag() = im;

        return out;
    }
    else
    {
        const size_t m = x.size();
        const size_t M = n.value_or(m);  // Output size
        if (M == 0)
            throw std::invalid_argument("Invalid DST length");
        const size_t t = forward ? type : (type == 2 ? 3 : type == 3 ? 2 : type);
        const bool ortho = orthogonalize.value_or(norm == Norm::ortho);
        const size_t ndst = type == 1 ? 2*(M+1) : 2*M;  // Normalize by DST length
        const R fct = scale<R>(norm, ndst, forward);

        InputView1D<Derived, R> in(x, M);  // Zero-pad or crop
        Vector<R, Dynamic> out(M);  // Output follows the input type
        const pocketfft::shape_t shape = {M};
        const pocketfft::shape_t axes = {axis};
        const pocketfft::stride_t stride_out = {static_cast<std::ptrdiff_t>(sizeof(R))};
        pocketfft::dst(shape, {in.stride()}, stride_out, axes, t, in.ptr(), out.data(), fct, ortho, workers);

        return out;
    }
}

template <bool forward, typename Derived>
auto pfft::core::dst_1d_extrude(const MatrixBase<Derived>& x, size_t type, std::optional<size_t> n, size_t axis,
                                Norm norm, size_t workers, std::optional<bool> orthogonalize)
{
    using S = Derived::Scalar;
    if (axis > 1)
        throw std::out_of_range("Entries of axis must be 0 or 1");

    const size_t r = x.rows(), c = x.cols();
    const size_t M = axis == 0 ? n.value_or(r) : r;
    const size_t N = axis == 1 ? n.value_or(c) : c;
    if (M == 0 || N == 0)
        throw std::invalid_argument("Invalid FFT length");

    Matrix<S, Dynamic, Dynamic> out(M, N);
    if (axis == 0)
        for (size_t j = 0; j < N; j++)
            out.col(j) = core::dst_1d<forward>(x.col(j), type, n, 0, norm, workers, orthogonalize);
    else
        for (size_t i = 0; i < M; i++)
            out.row(i) = core::dst_1d<forward>(x.row(i), type, n, 0, norm, workers, orthogonalize);

    return out;
}

template <typename Derived>
auto pfft::fft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::c2c_1d<true>(x, n, axis, norm, nthreads);
    else
        return core::c2c_1d_extrude<true>(x, n, axis, norm, nthreads);
}

template <typename Derived>
auto pfft::ifft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::c2c_1d<false>(x, n, axis, norm, nthreads);
    else
        return core::c2c_1d_extrude<false>(x, n, axis, norm, nthreads);
}

template <typename Derived>
auto pfft::rfft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::r2c_1d(x, n, axis, norm, nthreads);
    else
    {
        using R = Derived::Scalar;
        static_assert(!NumTraits<R>::IsComplex, "rfft inputs must be real");
        if (axis > 1)
            throw std::out_of_range("Entries of axis must be 0 or 1");

        const size_t r = x.rows(), c = x.cols();
        const size_t M = axis == 0 ? n.value_or(r)/2+1 : r;
        const size_t N = axis == 1 ? n.value_or(c)/2+1 : c;
        if (M == 0 || N == 0)
            throw std::invalid_argument("Invalid FFT length");

        Matrix<std::complex<R>, Dynamic, Dynamic> out(M, N);
        if (axis == 0)
            for (size_t j = 0; j < N; j++)
                out.col(j) = core::r2c_1d(x.col(j), n, 0, norm, nthreads);
        else
            for (size_t i = 0; i < M; i++)
                out.row(i) = core::r2c_1d(x.row(i), n, 0, norm, nthreads);

        return out;
    }
}

template <typename Derived>
auto pfft::irfft(const MatrixBase<Derived>& x, std::optional<size_t> n, size_t axis, Norm norm, size_t nthreads)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::c2r_1d(x, n, axis, norm, nthreads);
    else
    {
        using R = NumTraits<typename Derived::Scalar>::Real;
        if (axis > 1)
            throw std::out_of_range("Entries of axis must be 0 or 1");

        const size_t r = x.rows(), c = x.cols();
        const size_t M = axis == 0 ? n.value_or(r < 2 ? 0 : 2*(r-1)) : r;
        const size_t N = axis == 1 ? n.value_or(c < 2 ? 0 : 2*(c-1)) : c;
        if (M == 0 || N == 0)
            throw std::invalid_argument("Invalid FFT length");

        Matrix<R, Dynamic, Dynamic> out(M, N);
        if (axis == 0)
            for (size_t j = 0; j < N; j++)
                out.col(j) = core::c2r_1d(x.col(j), n, 0, norm, nthreads);
        else
            for (size_t i = 0; i < M; i++)
                out.row(i) = core::c2r_1d(x.row(i), n, 0, norm, nthreads);

        return out;
    }
}

template <typename Derived>
auto pfft::fft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes, Norm norm,
                size_t nthreads)
{
    if (axes.empty())
        throw std::invalid_argument("axes input must not be empty");
    if (x.rows() == 1 && x.cols() == 1)
        return Matrix<std::complex<typename NumTraits<typename Derived::Scalar>::Real>, Dynamic, Dynamic>(x);

    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2c_2d<true>(x, opt, axes, norm, nthreads);
}

template <typename Derived>
auto pfft::ifft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes, Norm norm,
                 size_t nthreads)
{
    if (axes.empty())
        throw std::invalid_argument("axes input must not be empty");
    if (x.rows() == 1 && x.cols() == 1)
        return Matrix<std::complex<typename NumTraits<typename Derived::Scalar>::Real>, Dynamic, Dynamic>(x);

    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2c_2d<false>(x, opt, axes, norm, nthreads);
}

template <typename Derived>
auto pfft::rfft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes, Norm norm,
                 size_t nthreads)
{
    if (axes.empty())
        throw std::invalid_argument("axes input must not be empty");
    if (x.rows() == 1 && x.cols() == 1)
        return Matrix<std::complex<typename NumTraits<typename Derived::Scalar>::Real>, Dynamic, Dynamic>(x);

    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::r2c_2d(x, opt, axes, norm, nthreads);
}

template <typename Derived>
auto pfft::irfft2(const MatrixBase<Derived>& x, const pocketfft::shape_t& s, const pocketfft::shape_t& axes, Norm norm,
                  size_t nthreads)
{
    if (axes.empty())
        throw std::invalid_argument("axes input must not be empty");
    if (x.rows() == 1 && x.cols() == 1)
        return Matrix<typename NumTraits<typename Derived::Scalar>::Real, Dynamic, Dynamic>(x.real());

    std::optional<pocketfft::shape_t> opt;
    if (!s.empty())
        opt = s;

    return core::c2r_2d(x, opt, axes, norm, nthreads);
}

template <typename Derived>
auto pfft::dst(const MatrixBase<Derived>& x, size_t type, std::optional<size_t> n, size_t axis, Norm norm,
               size_t workers, std::optional<bool> orthogonalize)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::dst_1d<true>(x, type, n, axis, norm, workers, orthogonalize);
    else
        return core::dst_1d_extrude<true>(x, type, n, axis, norm, workers, orthogonalize);
}

template <typename Derived>
auto pfft::idst(const MatrixBase<Derived>& x, size_t type, std::optional<size_t> n, size_t axis, Norm norm,
                size_t workers, std::optional<bool> orthogonalize)
{
    if constexpr (Derived::IsVectorAtCompileTime)
        return core::dst_1d<false>(x, type, n, axis, norm, workers, orthogonalize);
    else
        return core::dst_1d_extrude<false>(x, type, n, axis, norm, workers, orthogonalize);
}
