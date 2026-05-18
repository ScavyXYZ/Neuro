#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>
#include <filesystem>
#include <fstream>
#include <memory>

#if defined(_MSC_VER)
#  define NEURO_INLINE       __forceinline
#  define NEURO_RESTRICT     __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define NEURO_INLINE       __attribute__((always_inline)) inline
#  define NEURO_RESTRICT     __restrict__
#else
#  define NEURO_INLINE       inline
#  define NEURO_RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
#  define NEURO_LIKELY(x)    __builtin_expect(!!(x), 1)
#  define NEURO_UNLIKELY(x)  __builtin_expect(!!(x), 0)
#else
#  define NEURO_LIKELY(x)    (x)
#  define NEURO_UNLIKELY(x)  (x)
#endif

#if defined(__AVX2__)
#  include <immintrin.h>
#  define NEURO_HAS_AVX2  1
#  define NEURO_SIMD_WIDTH 4          
#elif defined(__SSE2__) || (defined(_MSC_VER) && defined(_M_X64))
#  include <emmintrin.h>
#  include <pmmintrin.h>
#  define NEURO_HAS_SSE2  1
#  define NEURO_SIMD_WIDTH 2          
#else
#  define NEURO_SIMD_WIDTH 1
#endif

#if defined(NEURO_HAS_AVX2)
#  define NEURO_ALIGN_BYTES 32
#elif defined(NEURO_HAS_SSE2)
#  define NEURO_ALIGN_BYTES 16
#else
#  define NEURO_ALIGN_BYTES 8
#endif

#if defined(_MSC_VER)
#  define NEURO_ALIGNAS __declspec(align(NEURO_ALIGN_BYTES))
#else
#  define NEURO_ALIGNAS alignas(NEURO_ALIGN_BYTES)
#endif

constexpr std::size_t neuro_pad(std::size_t n) noexcept
{
    return (n + NEURO_SIMD_WIDTH - 1) & ~(std::size_t(NEURO_SIMD_WIDTH) - 1);
}

template<typename T, std::size_t Alignment = NEURO_ALIGN_BYTES>
struct AlignedAllocator
{
    using value_type = T;
    using size_type = std::size_t;

    AlignedAllocator() noexcept = default;
    template<class U> AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n)
    {
        if (n == 0) return nullptr;
        void* ptr = nullptr;
#if defined(_MSC_VER)
        ptr = _aligned_malloc(n * sizeof(T), Alignment);
        if (!ptr) throw std::bad_alloc{};
#else
        if (posix_memalign(&ptr, Alignment, n * sizeof(T)) != 0)
            throw std::bad_alloc{};
#endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* ptr, std::size_t) noexcept
    {
#if defined(_MSC_VER)
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    template<class U>
    struct rebind { using other = AlignedAllocator<U, Alignment>; };

    bool operator==(const AlignedAllocator&) const noexcept { return true; }
    bool operator!=(const AlignedAllocator&) const noexcept { return false; }
};

using AlignedVector = std::vector<double, AlignedAllocator<double>>;

namespace neuro_detail
{
    NEURO_INLINE double dot(const double* NEURO_RESTRICT a,
        const double* NEURO_RESTRICT b,
        std::size_t n) noexcept
    {
#if defined(NEURO_HAS_AVX2)
        __m256d acc0 = _mm256_setzero_pd();
        __m256d acc1 = _mm256_setzero_pd();
        std::size_t i = 0;
        for (; i + 8 <= n; i += 8)
        {
            acc0 = _mm256_fmadd_pd(_mm256_load_pd(a + i),
                _mm256_loadu_pd(b + i), acc0);
            acc1 = _mm256_fmadd_pd(_mm256_load_pd(a + i + 4),
                _mm256_loadu_pd(b + i + 4), acc1);
        }
        for (; i + 4 <= n; i += 4)
            acc0 = _mm256_fmadd_pd(_mm256_load_pd(a + i),
                _mm256_loadu_pd(b + i), acc0);
        __m256d s256 = _mm256_add_pd(acc0, acc1);
        __m128d lo = _mm256_castpd256_pd128(s256);
        __m128d hi = _mm256_extractf128_pd(s256, 1);
        __m128d s128 = _mm_add_pd(lo, hi);
        s128 = _mm_hadd_pd(s128, s128);
        double result;
        _mm_store_sd(&result, s128);
        for (; i < n; ++i) result += a[i] * b[i];
        return result;

#elif defined(NEURO_HAS_SSE2)
        __m128d acc0 = _mm_setzero_pd();
        __m128d acc1 = _mm_setzero_pd();
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4)
        {
            acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_load_pd(a + i),
                _mm_loadu_pd(b + i)));
            acc1 = _mm_add_pd(acc1, _mm_mul_pd(_mm_load_pd(a + i + 2),
                _mm_loadu_pd(b + i + 2)));
        }
        for (; i + 2 <= n; i += 2)
            acc0 = _mm_add_pd(acc0, _mm_mul_pd(_mm_load_pd(a + i),
                _mm_loadu_pd(b + i)));
        __m128d s = _mm_add_pd(acc0, acc1);
        s = _mm_hadd_pd(s, s);
        double result;
        _mm_store_sd(&result, s);
        for (; i < n; ++i) result += a[i] * b[i];
        return result;

#else
        double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4)
        {
            acc0 += a[i] * b[i];
            acc1 += a[i + 1] * b[i + 1];
            acc2 += a[i + 2] * b[i + 2];
            acc3 += a[i + 3] * b[i + 3];
        }
        for (; i < n; ++i) acc0 += a[i] * b[i];
        return acc0 + acc1 + acc2 + acc3;
#endif
    }

    NEURO_INLINE void fma_add(double* NEURO_RESTRICT       out,
        const double* NEURO_RESTRICT  in,
        double                        scale,
        std::size_t                   n) noexcept
    {
#if defined(NEURO_HAS_AVX2)
        __m256d vs = _mm256_set1_pd(scale);
        std::size_t i = 0;
        for (; i + 4 <= n; i += 4)
            _mm256_store_pd(out + i,
                _mm256_fmadd_pd(vs, _mm256_loadu_pd(in + i),
                    _mm256_load_pd(out + i)));
        for (; i < n; ++i) out[i] += scale * in[i];

#elif defined(NEURO_HAS_SSE2)
        __m128d vs = _mm_set1_pd(scale);
        std::size_t i = 0;
        for (; i + 2 <= n; i += 2)
            _mm_store_pd(out + i,
                _mm_add_pd(_mm_load_pd(out + i),
                    _mm_mul_pd(vs, _mm_loadu_pd(in + i))));
        for (; i < n; ++i) out[i] += scale * in[i];

#else
        for (std::size_t i = 0; i < n; ++i)
            out[i] += scale * in[i];
#endif
    }

    using limit_func = std::pair<double, double>(*)(std::size_t, std::size_t, double);
    NEURO_INLINE std::pair<double, double> xavier_limit(std::size_t fan_in, std::size_t fan_out, double /*unused_a*/ = 0.0) noexcept
    {
        double limit = std::sqrt(6.0 / static_cast<double>(fan_in + fan_out));
        return { -limit, limit };
    }

    NEURO_INLINE std::pair<double, double> kaiming_limit(std::size_t fan_in, std::size_t /*unused_fan_out*/, double a = 0.0) noexcept
    {
        double gain = std::sqrt(2.0 / (1.0 + a * a));
        double limit = gain * std::sqrt(3.0 / static_cast<double>(fan_in));
        return { -limit, limit };
    }
}

struct activation_func
{
    double (*f)(double);
    double (*df)(double /*z*/, double /*a*/);
};

namespace activations {
    static constexpr activation_func ReLU = {
        .f = [](double x)         noexcept -> double { return x > 0.0 ? x : 0.0; },
        .df = [](double z, double) noexcept -> double { return z > 0.0 ? 1.0 : 0.0; }
    };
    static constexpr activation_func Sigmoid = {
        .f = [](double x)         noexcept -> double { return 1.0 / (1.0 + std::exp(-x)); },
        .df = [](double, double a) noexcept -> double { return a * (1.0 - a); }   // exact via a
    };
    static constexpr activation_func Tanh = {
        .f = [](double x)         noexcept -> double { return std::tanh(x); },
        .df = [](double, double a) noexcept -> double { return 1.0 - a * a; }     // exact via a
    };
    static constexpr activation_func LeakyReLU = {
        .f = [](double x)         noexcept -> double { return x > 0.0 ? x : 0.01 * x; },
        .df = [](double z, double) noexcept -> double { return z > 0.0 ? 1.0 : 0.01; }
    };
    static constexpr activation_func ELU = {
        .f = [](double x)         noexcept -> double { return x > 0.0 ? x : 0.01 * (std::exp(x) - 1.0); },
        .df = [](double z, double) noexcept -> double { return z > 0.0 ? 1.0 : 0.01 * std::exp(z); }
    };
    static constexpr activation_func SELU = {
        .f = [](double x) noexcept -> double {
            constexpr double lambda = 1.0507009873554804934193349852946;
            constexpr double alpha = 1.6732632423543772848170429916717;
            return x > 0.0 ? lambda * x : lambda * alpha * (std::exp(x) - 1.0);
        },
        .df = [](double z, double) noexcept -> double {
            constexpr double lambda = 1.0507009873554804934193349852946;
            constexpr double alpha = 1.6732632423543772848170429916717;
            return z > 0.0 ? lambda : lambda * alpha * std::exp(z);
        }
    };
    static constexpr activation_func GELU = {
        .f = [](double x) noexcept -> double {
            return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0)));
        },
        .df = [](double z, double) noexcept -> double {
            double cdf = 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
            double pdf = std::exp(-0.5 * z * z) / std::sqrt(2.0 * 3.14159265358979323846);
            return cdf + z * pdf;
        }
    };
    static constexpr activation_func Softplus = {
        .f = [](double x)         noexcept -> double { return std::log(1.0 + std::exp(x)); },
        .df = [](double, double a) noexcept -> double { return 1.0 - std::exp(-a); }  // exact via a
    };
    static constexpr activation_func Swish = {
        .f = [](double x)         noexcept -> double { return x / (1.0 + std::exp(-x)); },
        .df = [](double z, double) noexcept -> double {                                // exact via z
            double sig = 1.0 / (1.0 + std::exp(-z));
            return sig + z * sig * (1.0 - sig);
        }
    };
    static constexpr activation_func Mish = {
        .f = [](double x) noexcept -> double {
            return x * std::tanh(std::log(1.0 + std::exp(x)));
        },
        .df = [](double z, double) noexcept -> double {                                // exact via z
            double sp = std::log(1.0 + std::exp(z));
            double tanh_sp = std::tanh(sp);
            double sech_sp = 1.0 / std::cosh(sp);
            return tanh_sp + z * sech_sp * sech_sp * (std::exp(z) / (1.0 + std::exp(z)));
        }
    };
    static constexpr activation_func ReLU6 = {
        .f = [](double x)         noexcept -> double { return x > 0.0 ? (x < 6.0 ? x : 6.0) : 0.0; },
        .df = [](double z, double) noexcept -> double { return (z > 0.0 && z < 6.0) ? 1.0 : 0.0; }
    };
    static constexpr activation_func PReLU = {
        .f = [](double x)         noexcept -> double { return x > 0.0 ? x : 0.25 * x; },
        .df = [](double z, double) noexcept -> double { return z > 0.0 ? 1.0 : 0.25; }
    };
}

template<std::size_t InputSize, std::size_t... LayerSizes>
class Static_neuro
{
    static_assert(InputSize > 0, "Static_neuro: InputSize must be > 0.");
    static_assert(sizeof...(LayerSizes) > 0, "Static_neuro: at least one layer required.");

public:
    static constexpr std::size_t kNumLayers = sizeof...(LayerSizes);
    static constexpr std::size_t kInputSize = InputSize;
    static constexpr std::size_t kOutputSize = std::array<std::size_t, kNumLayers>{ LayerSizes... }[kNumLayers - 1];
    static constexpr std::size_t kSimdWidth = NEURO_SIMD_WIDTH;
    static constexpr std::size_t kAlignBytes = NEURO_ALIGN_BYTES;

private:
    static constexpr std::array<std::size_t, kNumLayers> kSizes = { LayerSizes... };

    static constexpr auto make_fan_in() noexcept
    {
        std::array<std::size_t, kNumLayers> fi{};
        fi[0] = InputSize;
        for (std::size_t i = 1; i < kNumLayers; ++i)
            fi[i] = kSizes[i - 1];
        return fi;
    }
    static constexpr std::array<std::size_t, kNumLayers> kFanIn = make_fan_in();

    static constexpr auto make_padded_fan_in() noexcept
    {
        std::array<std::size_t, kNumLayers> pfi{};
        for (std::size_t i = 0; i < kNumLayers; ++i)
            pfi[i] = neuro_pad(kFanIn[i]);
        return pfi;
    }
    static constexpr std::array<std::size_t, kNumLayers> kPaddedFanIn = make_padded_fan_in();

    static constexpr auto make_w_offset() noexcept
    {
        std::array<std::size_t, kNumLayers> off{};
        std::size_t acc = 0;
        for (std::size_t i = 0; i < kNumLayers; ++i)
        {
            off[i] = acc;
            acc += kSizes[i] * kPaddedFanIn[i];
        }
        return off;
    }
    static constexpr std::array<std::size_t, kNumLayers> kWOffset = make_w_offset();

    static constexpr std::size_t kTotalWeights = []() constexpr {
        std::size_t t = 0;
        constexpr auto pfi = make_padded_fan_in();
        for (std::size_t i = 0; i < kNumLayers; ++i)
            t += kSizes[i] * pfi[i];
        return t;
        }();

    static constexpr std::size_t kTotalNeurons = (LayerSizes + ...);

    static constexpr auto make_n_offset() noexcept
    {
        std::array<std::size_t, kNumLayers> off{};
        std::size_t acc = 0;
        for (std::size_t i = 0; i < kNumLayers; ++i) { off[i] = acc; acc += kSizes[i]; }
        return off;
    }
    static constexpr std::array<std::size_t, kNumLayers> kNOffset = make_n_offset();

    NEURO_INLINE double* row(std::size_t l, std::size_t j)       noexcept { return &weights_[kWOffset[l] + j * kPaddedFanIn[l]]; }
    NEURO_INLINE const double* row(std::size_t l, std::size_t j) const noexcept { return &weights_[kWOffset[l] + j * kPaddedFanIn[l]]; }
    NEURO_INLINE double* aptr(std::size_t l)       noexcept { return &activations_[kNOffset[l]]; }
    NEURO_INLINE const double* aptr(std::size_t l) const noexcept { return &activations_[kNOffset[l]]; }
    NEURO_INLINE double* zptr(std::size_t l)       noexcept { return &pre_activations_[kNOffset[l]]; }
    NEURO_INLINE double* dptr(std::size_t l)       noexcept { return &deltas_[kNOffset[l]]; }
    NEURO_INLINE const double* dptr(std::size_t l) const noexcept { return &deltas_[kNOffset[l]]; }

public:
    using ForwardFn = double(*)(double);
    using DerivFn = double(*)(double, double);

    Static_neuro(const activation_func& actv, double lr = 0.01) noexcept
        : activation_(actv.f), activation_d_(actv.df), lr_(lr)
    {
        weights_.fill(0.0);
        biases_.fill(0.0);
        activations_.fill(0.0);
        pre_activations_.fill(0.0);
        deltas_.fill(0.0);
    }

    static constexpr std::size_t input_size()    noexcept { return kInputSize; }
    static constexpr std::size_t num_layers()    noexcept { return kNumLayers; }
    static constexpr std::size_t output_size()   noexcept { return kOutputSize; }
    static constexpr std::size_t total_weights() noexcept { return kTotalWeights; }
    static constexpr std::size_t simd_width()    noexcept { return kSimdWidth; }
    static constexpr std::size_t align_bytes()   noexcept { return kAlignBytes; }

    double learning_rate()              const noexcept { return lr_; }
    void   set_learning_rate(double lr)       noexcept { lr_ = lr; }

    void init(neuro_detail::limit_func limit_func, std::uint32_t seed = 0, double a = 0.0)
    {
        std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);

        for (std::size_t i = 0; i < kNumLayers; ++i)
        {
            std::pair<double, double> bounds = limit_func(kFanIn[i], kSizes[i], a);

            std::uniform_real_distribution<double> dist(bounds.first, bounds.second);

            for (std::size_t j = 0; j < kSizes[i]; ++j)
            {
                double* w = row(i, j);
                for (std::size_t k = 0; k < kFanIn[i]; ++k)               w[k] = dist(rng);
                for (std::size_t k = kFanIn[i]; k < kPaddedFanIn[i]; ++k) w[k] = 0.0;
            }
            std::fill_n(&biases_[kNOffset[i]], kSizes[i], 0.0);
        }
    }

    bool save(const std::filesystem::path& path) const noexcept
    {
        try {
            std::ofstream file(path, std::ios::binary);
            if (!file.is_open()) return false;
            for (std::size_t i = 0; i < kNumLayers; ++i)
                for (std::size_t j = 0; j < kSizes[i]; ++j)
                    file.write(reinterpret_cast<const char*>(row(i, j)),
                        static_cast<std::streamsize>(kFanIn[i] * sizeof(double)));
            file.write(reinterpret_cast<const char*>(biases_.data()),
                static_cast<std::streamsize>(kTotalNeurons * sizeof(double)));
            return file.good();
        }
        catch (...) { return false; }
    }

    bool load(const std::filesystem::path& path) noexcept
    {
        try {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) return false;
            for (std::size_t i = 0; i < kNumLayers; ++i)
            {
                for (std::size_t j = 0; j < kSizes[i]; ++j)
                {
                    double* w = row(i, j);
                    file.read(reinterpret_cast<char*>(w),
                        static_cast<std::streamsize>(kFanIn[i] * sizeof(double)));
                    for (std::size_t k = kFanIn[i]; k < kPaddedFanIn[i]; ++k) w[k] = 0.0;
                }
            }
            file.read(reinterpret_cast<char*>(biases_.data()),
                static_cast<std::streamsize>(kTotalNeurons * sizeof(double)));
            return file.good();
        }
        catch (...) { return false; }
    }

    [[nodiscard]]
    std::span<const double, kOutputSize> predict(std::span<const double, InputSize> input) noexcept
    {
        forward(input.data());
        return std::span<const double, kOutputSize>(aptr(kNumLayers - 1), kOutputSize);
    }

    [[nodiscard]]
    std::span<const double, kOutputSize> predict(const std::array<double, InputSize>& input) noexcept
    {
        return predict(std::span<const double, InputSize>(input));
    }

    void train(std::span<const double, InputSize>   input,
        std::span<const double, kOutputSize> target) noexcept
    {
        forward(input.data());
        backward(input.data(), target.data());
    }

    void train(const std::array<double, InputSize>& input,
        const std::array<double, kOutputSize>& target) noexcept
    {
        train(std::span<const double, InputSize>(input),
            std::span<const double, kOutputSize>(target));
    }

private:
    NEURO_INLINE void forward(const double* NEURO_RESTRICT input) noexcept
    {
        [&] <std::size_t... Is>(std::index_sequence<Is...>) noexcept
        {
            (forward_layer<Is>(input), ...);
        }
        (std::make_index_sequence<kNumLayers>{});
    }

    template<std::size_t I>
    NEURO_INLINE void forward_layer(const double* NEURO_RESTRICT input) noexcept
    {
        const double* NEURO_RESTRICT prev = (I == 0) ? input : aptr(I - 1);
        double* NEURO_RESTRICT z = zptr(I);
        double* NEURO_RESTRICT a = aptr(I);
        const double* NEURO_RESTRICT b = &biases_[kNOffset[I]];
        const std::size_t dot_n = (I == 0) ? kFanIn[I] : kPaddedFanIn[I];
        for (std::size_t j = 0; j < kSizes[I]; ++j)
        {
            z[j] = b[j] + neuro_detail::dot(row(I, j), prev, dot_n);
            a[j] = activation_(z[j]);
        }
    }

    NEURO_INLINE void backward(const double* NEURO_RESTRICT input,
        const double* NEURO_RESTRICT target) noexcept
    {
        {
            constexpr std::size_t L = kNumLayers - 1;
            double* NEURO_RESTRICT d = dptr(L);
            const double* NEURO_RESTRICT a = aptr(L);
            const double* NEURO_RESTRICT z = zptr(L);
            for (std::size_t j = 0; j < kSizes[L]; ++j)
                d[j] = activation_d_(z[j], a[j]) * (target[j] - a[j]);
        }
        [&] <std::size_t... Is>(std::index_sequence<Is...>) noexcept
        {
            (backward_hidden_layer<kNumLayers - 2 - Is>(), ...);
        }
        (std::make_index_sequence<kNumLayers - 1>{});

        [&] <std::size_t... Is>(std::index_sequence<Is...>) noexcept
        {
            (update_layer<Is>(input), ...);
        }
        (std::make_index_sequence<kNumLayers>{});
    }

    template<std::size_t I>
    NEURO_INLINE void backward_hidden_layer() noexcept
    {
        double* NEURO_RESTRICT d_curr = dptr(I);
        const double* NEURO_RESTRICT d_next = dptr(I + 1);
        const double* NEURO_RESTRICT a_curr = aptr(I);
        const double* NEURO_RESTRICT z_curr = zptr(I);
        for (std::size_t j = 0; j < kSizes[I]; ++j)
        {
            double sum = 0.0;
            for (std::size_t k = 0; k < kSizes[I + 1]; ++k)
                sum += d_next[k] * row(I + 1, k)[j];
            d_curr[j] = activation_d_(z_curr[j], a_curr[j]) * sum;
        }
    }

    template<std::size_t I>
    NEURO_INLINE void update_layer(const double* NEURO_RESTRICT input) noexcept
    {
        const double* NEURO_RESTRICT prev = (I == 0) ? input : aptr(I - 1);
        const double* NEURO_RESTRICT d = dptr(I);
        double* NEURO_RESTRICT b = &biases_[kNOffset[I]];
        const std::size_t fma_n = (I == 0) ? kFanIn[I] : kPaddedFanIn[I];
        for (std::size_t j = 0; j < kSizes[I]; ++j)
        {
            const double lr_d = lr_ * d[j];
            b[j] += lr_d;
            neuro_detail::fma_add(row(I, j), prev, lr_d, fma_n);
        }
    }

    ForwardFn    activation_;
    DerivFn      activation_d_;
    double       lr_;

    NEURO_ALIGNAS std::array<double, kTotalWeights> weights_;
    NEURO_ALIGNAS std::array<double, kTotalNeurons> biases_;
    NEURO_ALIGNAS std::array<double, kTotalNeurons> activations_;
    NEURO_ALIGNAS std::array<double, kTotalNeurons> pre_activations_;
    NEURO_ALIGNAS std::array<double, kTotalNeurons> deltas_;
};

class Neuro
{
public:
    using ForwardFn = double(*)(double);
    using DerivFn = double(*)(double, double);

    Neuro() : input_size_(0), lr_(0.01), n_layers_(0) {}

    Neuro(std::size_t                    input_size,
        const std::vector<std::size_t>& layer_sizes,
        const activation_func& actv,
        double                          lr = 0.01)
        : input_size_(input_size)
        , lr_(lr)
        , activation_(actv.f)
        , activation_d_(actv.df)
        , layer_sizes_(layer_sizes)
        , n_layers_(layer_sizes.size())
    {
        if (NEURO_UNLIKELY(layer_sizes.empty()))
            throw std::invalid_argument("Neuro: layer_sizes must not be empty.");
        rebuild_offsets_and_buffers();
    }

    static Neuro from_file(const std::filesystem::path& path,
        const activation_func& actv)
    {
        Neuro net;
        net.activation_ = actv.f;
        net.activation_d_ = actv.df;
        if (!net.load(path))
            throw std::runtime_error(
                "Neuro::from_file: cannot load '" + path.string() + "'");
        return net;
    }

    void set_activation(const activation_func& actv)
    {
        activation_ = actv.f;
        activation_d_ = actv.df;
    }

    bool save(const std::filesystem::path& path) const noexcept
    {
        try {
            std::ofstream file(path, std::ios::binary);
            if (!file.is_open()) return false;
            file.write(reinterpret_cast<const char*>(&input_size_), sizeof(input_size_));
            file.write(reinterpret_cast<const char*>(&lr_), sizeof(lr_));
            file.write(reinterpret_cast<const char*>(&n_layers_), sizeof(n_layers_));
            file.write(reinterpret_cast<const char*>(layer_sizes_.data()),
                static_cast<std::streamsize>(n_layers_ * sizeof(std::size_t)));
            for (std::size_t i = 0; i < n_layers_; ++i)
            {
                const std::size_t fan_in = (i == 0) ? input_size_ : layer_sizes_[i - 1];
                for (std::size_t j = 0; j < layer_sizes_[i]; ++j)
                    file.write(reinterpret_cast<const char*>(wrow(i, j)),
                        static_cast<std::streamsize>(fan_in * sizeof(double)));
            }
            file.write(reinterpret_cast<const char*>(biases_.data()),
                static_cast<std::streamsize>(biases_.size() * sizeof(double)));
            return file.good();
        }
        catch (...) { return false; }
    }

    bool load(const std::filesystem::path& path) noexcept
    {
        try {
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) return false;
            file.read(reinterpret_cast<char*>(&input_size_), sizeof(input_size_));
            file.read(reinterpret_cast<char*>(&lr_), sizeof(lr_));
            file.read(reinterpret_cast<char*>(&n_layers_), sizeof(n_layers_));
            layer_sizes_.resize(n_layers_);
            file.read(reinterpret_cast<char*>(layer_sizes_.data()),
                static_cast<std::streamsize>(n_layers_ * sizeof(std::size_t)));
            rebuild_offsets_and_buffers();
            for (std::size_t i = 0; i < n_layers_; ++i)
            {
                const std::size_t fan_in = (i == 0) ? input_size_ : layer_sizes_[i - 1];
                const std::size_t pad_fi = padded_fan_in(i);
                for (std::size_t j = 0; j < layer_sizes_[i]; ++j)
                {
                    double* w = wrow(i, j);
                    file.read(reinterpret_cast<char*>(w),
                        static_cast<std::streamsize>(fan_in * sizeof(double)));
                    for (std::size_t k = fan_in; k < pad_fi; ++k) w[k] = 0.0;
                }
            }
            file.read(reinterpret_cast<char*>(biases_.data()),
                static_cast<std::streamsize>(biases_.size() * sizeof(double)));
            return file.good();
        }
        catch (...) { return false; }
    }

    std::size_t input_size()              const noexcept { return input_size_; }
    std::size_t num_layers()              const noexcept { return n_layers_; }
    std::size_t output_size()             const noexcept { return layer_sizes_.back(); }
    std::size_t layer_size(std::size_t i) const { return layer_sizes_.at(i); }
    double      learning_rate()           const noexcept { return lr_; }
    void        set_learning_rate(double lr)    noexcept { lr_ = lr; }
    bool        ready()                   const noexcept { return n_layers_ > 0 && input_size_ > 0; }

    void init(neuro_detail::limit_func limit_fn, std::uint32_t seed = 0, double a = 0.0)
    {
        std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);

        for (std::size_t i = 0; i < n_layers_; ++i)
        {
            const std::size_t fan_in = (i == 0) ? input_size_ : layer_sizes_[i - 1];
            const std::size_t pad_fi = padded_fan_in(i);

            const auto [limit_min, limit_max] = limit_fn(fan_in, layer_sizes_[i], a);
            std::uniform_real_distribution<double> dist(limit_min, limit_max);

            for (std::size_t j = 0; j < layer_sizes_[i]; ++j)
            {
                double* w = wrow(i, j);
                for (std::size_t k = 0; k < fan_in; ++k)      w[k] = dist(rng);
                for (std::size_t k = fan_in; k < pad_fi; ++k) w[k] = 0.0;
            }
            std::fill_n(&biases_[n_offset_[i]], layer_sizes_[i], 0.0);
        }
    }

    void init(const std::vector<std::pair<double, double>>& ranges, std::uint32_t seed = 0)
    {
        if (NEURO_UNLIKELY(ranges.size() != n_layers_))
            throw std::invalid_argument(
                "Neuro::init: ranges.size() (" + std::to_string(ranges.size()) +
                ") must equal layer count (" + std::to_string(n_layers_) + ").");
        std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
        for (std::size_t i = 0; i < n_layers_; ++i)
        {
            const std::size_t fan_in = (i == 0) ? input_size_ : layer_sizes_[i - 1];
            const std::size_t pad_fi = padded_fan_in(i);
            std::uniform_real_distribution<double> dist(ranges[i].first, ranges[i].second);
            for (std::size_t j = 0; j < layer_sizes_[i]; ++j)
            {
                double* w = wrow(i, j);
                for (std::size_t k = 0; k < fan_in; ++k)      w[k] = dist(rng);
                for (std::size_t k = fan_in; k < pad_fi; ++k) w[k] = 0.0;
            }
            std::fill_n(&biases_[n_offset_[i]], layer_sizes_[i], 0.0);
        }
    }

    [[nodiscard]]
    std::span<const double> predict(std::span<const double> input)
    {
        check_ready("predict");
        check_size(input.size(), input_size_, "predict", "input");
        forward(input.data());
        return { &activations_[n_offset_.back()], layer_sizes_.back() };
    }

    [[nodiscard]]
    std::span<const double> predict(const std::vector<double>& input)
    {
        return predict(std::span<const double>(input));
    }

    void train(std::span<const double> input, std::span<const double> target)
    {
        check_ready("train");
        check_size(input.size(), input_size_, "train", "input");
        check_size(target.size(), output_size(), "train", "target");
        forward(input.data());
        backward(input.data(), target.data());
    }

    void train(const std::vector<double>& input, const std::vector<double>& target)
    {
        train(std::span<const double>(input), std::span<const double>(target));
    }

private:
    NEURO_INLINE std::size_t padded_fan_in(std::size_t i) const noexcept
    {
        return neuro_pad((i == 0) ? input_size_ : layer_sizes_[i - 1]);
    }

    void rebuild_offsets_and_buffers()
    {
        w_offset_.resize(n_layers_);
        n_offset_.resize(n_layers_);
        std::size_t w_total = 0, n_total = 0;
        for (std::size_t i = 0; i < n_layers_; ++i)
        {
            w_offset_[i] = w_total;
            n_offset_[i] = n_total;
            w_total += layer_sizes_[i] * padded_fan_in(i);
            n_total += layer_sizes_[i];
        }
        weights_.assign(w_total, 0.0);
        biases_.assign(n_total, 0.0);
        activations_.assign(n_total, 0.0);
        pre_activations_.assign(n_total, 0.0);
        deltas_.assign(n_total, 0.0);
    }

    NEURO_INLINE double* wrow(std::size_t i, std::size_t j) noexcept
    {
        return &weights_[w_offset_[i] + j * padded_fan_in(i)];
    }
    NEURO_INLINE const double* wrow(std::size_t i, std::size_t j) const noexcept
    {
        return &weights_[w_offset_[i] + j * padded_fan_in(i)];
    }

    NEURO_INLINE void forward(const double* NEURO_RESTRICT input) noexcept
    {
        for (std::size_t i = 0; i < n_layers_; ++i)
        {
            const double* NEURO_RESTRICT prev =
                (i == 0) ? input : &activations_[n_offset_[i - 1]];
            const std::size_t fan_in = (i == 0) ? input_size_ : layer_sizes_[i - 1];
            const std::size_t dot_n = (i == 0) ? fan_in : padded_fan_in(i);
            const std::size_t n = layer_sizes_[i];
            double* NEURO_RESTRICT z = &pre_activations_[n_offset_[i]];
            double* NEURO_RESTRICT a = &activations_[n_offset_[i]];
            const double* NEURO_RESTRICT b = &biases_[n_offset_[i]];
            for (std::size_t j = 0; j < n; ++j)
            {
                z[j] = b[j] + neuro_detail::dot(wrow(i, j), prev, dot_n);
                a[j] = activation_(z[j]);
            }
        }
    }

    NEURO_INLINE void backward(const double* NEURO_RESTRICT input,
        const double* NEURO_RESTRICT target) noexcept
    {
        {
            const std::size_t L = n_layers_ - 1;
            const std::size_t n = layer_sizes_[L];
            double* NEURO_RESTRICT d = &deltas_[n_offset_[L]];
            const double* NEURO_RESTRICT a = &activations_[n_offset_[L]];
            const double* NEURO_RESTRICT z = &pre_activations_[n_offset_[L]];
            for (std::size_t j = 0; j < n; ++j)
                d[j] = activation_d_(z[j], a[j]) * (target[j] - a[j]);
        }
        for (std::size_t ii = n_layers_; ii-- > 1; )
        {
            const std::size_t i = ii - 1;
            const std::size_t i_next = ii;
            const std::size_t n_curr = layer_sizes_[i];
            const std::size_t n_next = layer_sizes_[i_next];
            double* NEURO_RESTRICT d_curr = &deltas_[n_offset_[i]];
            const double* NEURO_RESTRICT d_next = &deltas_[n_offset_[i_next]];
            const double* NEURO_RESTRICT a_curr = &activations_[n_offset_[i]];
            const double* NEURO_RESTRICT z_curr = &pre_activations_[n_offset_[i]];
            for (std::size_t j = 0; j < n_curr; ++j)
            {
                double sum = 0.0;
                for (std::size_t k = 0; k < n_next; ++k)
                    sum += d_next[k] * wrow(i_next, k)[j];
                d_curr[j] = activation_d_(z_curr[j], a_curr[j]) * sum;
            }
        }
        for (std::size_t i = 0; i < n_layers_; ++i)
        {
            const double* NEURO_RESTRICT prev =
                (i == 0) ? input : &activations_[n_offset_[i - 1]];
            const std::size_t fan_in = (i == 0) ? input_size_ : layer_sizes_[i - 1];
            const std::size_t fma_n = (i == 0) ? fan_in : padded_fan_in(i);
            const std::size_t n = layer_sizes_[i];
            const double* NEURO_RESTRICT d = &deltas_[n_offset_[i]];
            double* NEURO_RESTRICT b = &biases_[n_offset_[i]];
            for (std::size_t j = 0; j < n; ++j)
            {
                const double lr_d = lr_ * d[j];
                b[j] += lr_d;
                neuro_detail::fma_add(wrow(i, j), prev, lr_d, fma_n);
            }
        }
    }

    NEURO_INLINE void check_size(std::size_t got, std::size_t expected,
        const char* fn, const char* arg) const
    {
        assert(got == expected);
        if (NEURO_UNLIKELY(got != expected))
            throw std::invalid_argument(
                std::string("Neuro::") + fn + ": " + arg +
                " size mismatch - expected " + std::to_string(expected) +
                ", got " + std::to_string(got) + ".");
    }

    NEURO_INLINE void check_ready(const char* fn) const
    {
        if (NEURO_UNLIKELY(!ready()))
            throw std::logic_error(
                std::string("Neuro::") + fn +
                ": network not initialized. Call load() or use the full constructor.");
    }

    std::size_t              input_size_;
    double                   lr_;
    ForwardFn                activation_;
    DerivFn                  activation_d_;
    std::vector<std::size_t> layer_sizes_;
    std::size_t              n_layers_;
    std::vector<std::size_t> w_offset_;
    std::vector<std::size_t> n_offset_;

    AlignedVector weights_;
    AlignedVector biases_;
    AlignedVector activations_;
    AlignedVector pre_activations_;
    AlignedVector deltas_;
};