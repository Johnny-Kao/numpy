// Research-only selector microbenchmark for NumPy PR #32895.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

static constexpr std::size_t Q = 1u << 20;
static constexpr std::size_t N = 1u << 20;
static constexpr int SAMPLES = 16;
static constexpr int REPEATS = 200000;

static inline std::size_t anchor(int j) {
    return (static_cast<std::size_t>(j) * (Q - 1)) >> 4;
}

__attribute__((noinline))
bool current_anchors(const int32_t* q) {
    int direction = 0;
    int32_t prev = q[0];
    for (int j = 1; j <= SAMPLES; ++j) {
        const int32_t cur = q[anchor(j)];
        if (cur > prev) {
            if (direction < 0) return false;
            direction = 1;
        } else if (cur < prev) {
            if (direction > 0) return false;
            direction = -1;
        }
        prev = cur;
    }
    return direction >= 0;
}

__attribute__((noinline))
bool adjacent_monotone(const int32_t* q) {
    // Keep the existing anchor check, then inspect immediate neighborhoods.
    if (!current_anchors(q)) return false;
    for (int j = 0; j <= SAMPLES; ++j) {
        const std::size_t i = anchor(j);
        if (i > 0 && q[i] < q[i - 1]) return false;
        if (i + 1 < Q && q[i + 1] < q[i]) return false;
    }
    return true;
}

__attribute__((noinline))
bool adjacent_signflip(const int32_t* q) {
    if (!current_anchors(q)) return false;
    int last_sign = 0;
    for (int j = 0; j <= SAMPLES; ++j) {
        const std::size_t i = anchor(j);
        if (i == 0 || i + 1 >= Q) continue;
        const int32_t a = q[i - 1], b = q[i], c = q[i + 1];
        const int s1 = (b > a) - (b < a);
        const int s2 = (c > b) - (c < b);
        if (s1 < 0 || s2 < 0) return false;
        const int s = s2 ? s2 : s1;
        if (s && last_sign && s != last_sign) return false;
        if (s) last_sign = s;
    }
    return true;
}

__attribute__((noinline))
bool adjacent_roughness_i64(const int32_t* q) {
    if (!current_anchors(q)) return false;
    int64_t s1 = 0, s2 = 0, prev_d = 0;
    bool have_prev = false;
    for (int j = 0; j <= SAMPLES; ++j) {
        const std::size_t i = anchor(j);
        if (i == 0 || i + 1 >= Q) continue;
        const int64_t a = q[i - 1], b = q[i], c = q[i + 1];
        const int64_t d1 = b - a;
        const int64_t d2 = c - b;
        if (d1 < 0 || d2 < 0) return false;
        s1 += std::llabs(d1) + std::llabs(d2);
        if (have_prev) s2 += std::llabs(d1 - prev_d);
        s2 += std::llabs(d2 - d1);
        prev_d = d2;
        have_prev = true;
    }
    // Conservative integer inequality; exact threshold is not the point here.
    return s2 <= 8 * std::max<int64_t>(1, s1);
}

__attribute__((noinline))
bool multiscale_monotone(const int32_t* q) {
    // 17 anchors + 16 midpoints = 33 deterministic probes.
    int32_t prev = q[0];
    for (int j = 0; j < SAMPLES; ++j) {
        const std::size_t lo = anchor(j);
        const std::size_t hi = anchor(j + 1);
        const std::size_t mid = lo + ((hi - lo) >> 1);
        const int32_t m = q[mid];
        const int32_t h = q[hi];
        if (m < prev || h < m) return false;
        prev = h;
    }
    return true;
}

template <class F>
double bench_ns(const int32_t* q, F fn, volatile std::uint64_t& sink) {
    for (int i = 0; i < 1000; ++i) sink += fn(q);
    std::array<double, 15> samples{};
    for (double& out : samples) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPEATS; ++i) sink += fn(q);
        const auto t1 = std::chrono::steady_clock::now();
        out = std::chrono::duration<double, std::nano>(t1 - t0).count() / REPEATS;
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

static void force_anchor_deception(std::vector<int32_t>& q) {
    for (int j = 0; j <= SAMPLES; ++j) {
        q[anchor(j)] = static_cast<int32_t>((static_cast<std::uint64_t>(j) * (N - 1)) / SAMPLES);
    }
}

int main() {
    std::vector<int32_t> ordered(Q), random(Q), oscillating(Q), reversed_blocks(Q);
    for (std::size_t i = 0; i < Q; ++i) ordered[i] = static_cast<int32_t>(i);

    std::mt19937 rng(20261007);
    std::uniform_int_distribution<int32_t> dist(0, N - 1);
    for (auto& x : random) x = dist(rng);
    for (std::size_t i = 0; i < Q; ++i) oscillating[i] = (i & 1) ? 0 : static_cast<int32_t>(N - 1);
    constexpr std::size_t block = 4096;
    for (std::size_t i = 0; i < Q; ++i) {
        const std::size_t base = i % N;
        const std::size_t block_id = i / block;
        reversed_blocks[i] = static_cast<int32_t>((block_id & 1) ? (N - 1 - base) : base);
    }
    force_anchor_deception(random);
    force_anchor_deception(oscillating);
    force_anchor_deception(reversed_blocks);

    struct Candidate { const char* name; bool (*fn)(const int32_t*); };
    const Candidate cs[] = {
        {"current_anchors", current_anchors},
        {"adjacent_monotone", adjacent_monotone},
        {"adjacent_signflip", adjacent_signflip},
        {"adjacent_roughness_i64", adjacent_roughness_i64},
        {"multiscale_monotone", multiscale_monotone},
    };
    struct Workload { const char* name; const std::vector<int32_t>* q; };
    const Workload ws[] = {
        {"ordered", &ordered},
        {"deception_random", &random},
        {"deception_oscillating", &oscillating},
        {"deception_reversed_blocks", &reversed_blocks},
    };

    volatile std::uint64_t sink = 0;
    std::puts("candidate,workload,accept,median_ns");
    for (const auto& c : cs) {
        for (const auto& w : ws) {
            const bool accept = c.fn(w.q->data());
            const double ns = bench_ns(w.q->data(), c.fn, sink);
            std::printf("%s,%s,%d,%.3f\n", c.name, w.name, accept ? 1 : 0, ns);
        }
    }
    std::fprintf(stderr, "sink=%llu\n", static_cast<unsigned long long>(sink));
}
