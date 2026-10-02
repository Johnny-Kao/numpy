#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

using i64 = std::int64_t;
using Clock = std::chrono::steady_clock;

static std::vector<i64>
current_batched(const std::vector<std::int32_t>& arr,
                const std::vector<std::int32_t>& keys)
{
    std::vector<i64> ret(keys.size(), 0);
    const i64 n = static_cast<i64>(arr.size());
    if (n <= 0) {
        return ret;
    }

    i64 interval_length = n;
    i64 half = interval_length >> 1;
    interval_length -= half;

    const std::int32_t mid_val = arr[half];
    for (std::size_t i = 0; i < keys.size(); ++i) {
        ret[i] = (mid_val < keys[i]) * half;
    }

    while (interval_length > 1) {
        half = interval_length >> 1;
        interval_length -= half;

        for (std::size_t i = 0; i < keys.size(); ++i) {
            i64& base = ret[i];
            const std::int32_t probe = arr[base + half];
            base += (probe < keys[i]) * half;
        }
    }

    for (std::size_t i = 0; i < keys.size(); ++i) {
        i64& base = ret[i];
        base += arr[base] < keys[i];
    }
    return ret;
}

static std::vector<i64>
previous_bound(const std::vector<std::int32_t>& arr,
               const std::vector<std::int32_t>& keys)
{
    std::vector<i64> ret(keys.size(), 0);
    if (keys.empty()) {
        return ret;
    }

    const i64 n = static_cast<i64>(arr.size());
    i64 min_idx = 0;
    i64 max_idx = n;
    std::int32_t last_key = keys[0];

    for (std::size_t q = 0; q < keys.size(); ++q) {
        const std::int32_t key = keys[q];

        if (last_key < key) {
            max_idx = n;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < n) ? max_idx + 1 : n;
        }
        last_key = key;

        while (min_idx < max_idx) {
            const i64 mid = min_idx + ((max_idx - min_idx) >> 1);
            if (arr[mid] < key) {
                min_idx = mid + 1;
            }
            else {
                max_idx = mid;
            }
        }
        ret[q] = min_idx;
    }
    return ret;
}

static std::vector<i64>
galloping(const std::vector<std::int32_t>& arr,
          const std::vector<std::int32_t>& keys)
{
    std::vector<i64> ret(keys.size(), 0);
    const i64 n = static_cast<i64>(arr.size());
    if (n <= 0 || keys.empty()) {
        return ret;
    }

    std::int32_t last_key = keys[0];
    i64 previous_pos = 0;

    for (std::size_t q = 0; q < keys.size(); ++q) {
        const std::int32_t key = keys[q];
        i64 min_idx = 0;
        i64 max_idx = n;

        if (q > 0 && key >= last_key) {
            min_idx = previous_pos;

            if (min_idx < n && arr[min_idx] < key) {
                const i64 origin = min_idx;
                i64 step = 1;
                min_idx = origin + 1;

                while (true) {
                    const i64 remaining = n - 1 - origin;
                    if (step > remaining) {
                        max_idx = n;
                        break;
                    }

                    const i64 probe = origin + step;
                    if (!(arr[probe] < key)) {
                        max_idx = probe + 1;
                        break;
                    }

                    min_idx = probe + 1;
                    if (step > (remaining >> 1)) {
                        max_idx = n;
                        break;
                    }
                    step <<= 1;
                }
            }
            else {
                max_idx = (min_idx < n) ? min_idx + 1 : n;
            }
        }

        while (min_idx < max_idx) {
            const i64 mid = min_idx + ((max_idx - min_idx) >> 1);
            if (arr[mid] < key) {
                min_idx = mid + 1;
            }
            else {
                max_idx = mid;
            }
        }

        ret[q] = min_idx;
        previous_pos = min_idx;
        last_key = key;
    }
    return ret;
}


static std::vector<i64>
hybrid_sorted_gate(const std::vector<std::int32_t>& arr,
                   const std::vector<std::int32_t>& keys)
{
    for (std::size_t i = 1; i < keys.size(); ++i) {
        if (keys[i] < keys[i - 1]) {
            return current_batched(arr, keys);
        }
    }
    return galloping(arr, keys);
}

static std::vector<std::int32_t>
make_fixed_delta(i64 n, i64 q, i64 delta)
{
    std::vector<std::int32_t> keys;
    keys.reserve(q);
    const i64 start = n / 10;
    if (q <= 0 || delta < 0 || start + (q - 1) * delta >= n) {
        return keys;
    }
    for (i64 i = 0; i < q; ++i) {
        keys.push_back(static_cast<std::int32_t>(start + i * delta));
    }
    return keys;
}

static std::vector<std::int32_t>
make_queries(i64 n, i64 q, const std::string& shape, std::uint64_t seed)
{
    std::vector<std::int32_t> keys;
    keys.reserve(q);
    std::mt19937_64 rng(seed);

    if (shape == "random") {
        std::uniform_int_distribution<i64> dist(-n / 10, n + n / 10);
        for (i64 i = 0; i < q; ++i) {
            keys.push_back(static_cast<std::int32_t>(dist(rng)));
        }
        return keys;
    }

    i64 pos = n / 10;
    for (i64 i = 0; i < q; ++i) {
        if (pos >= n) {
            return {};
        }
        keys.push_back(static_cast<std::int32_t>(pos));

        i64 step = 1;
        if (shape == "dense") {
            step = 1;
        }
        else if (shape == "medium") {
            step = 10;
        }
        else if (shape == "sparse") {
            step = std::max<i64>(100, n / std::max<i64>(q * 4, 1));
        }
        else if (shape == "clustered") {
            step = ((i + 1) % 5 == 0) ? 64 : 1;
        }
        else if (shape == "mostly_monotonic") {
            step = 1;
        }
        else {
            return {};
        }
        pos += step;
        if (shape == "mostly_monotonic" && (i + 1) % 32 == 0) {
            pos = std::max<i64>(0, pos - 16);
        }
    }
    return keys;
}

using Algo = std::vector<i64>(*)(const std::vector<std::int32_t>&,
                               const std::vector<std::int32_t>&);

static bool
verify(const std::vector<std::int32_t>& arr,
       const std::vector<std::int32_t>& keys,
       const std::vector<i64>& got)
{
    if (got.size() != keys.size()) {
        return false;
    }
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const i64 expected = std::lower_bound(arr.begin(), arr.end(), keys[i]) - arr.begin();
        if (got[i] != expected) {
            std::cerr << "mismatch i=" << i
                      << " key=" << keys[i]
                      << " expected=" << expected
                      << " got=" << got[i] << "\n";
            return false;
        }
    }
    return true;
}

static double
time_algo(Algo algo,
          const std::vector<std::int32_t>& arr,
          const std::vector<std::int32_t>& keys)
{
    constexpr int warmups = 2;
    constexpr int repeats = 7;

    volatile i64 sink = 0;
    for (int i = 0; i < warmups; ++i) {
        auto out = algo(arr, keys);
        sink = sink + out[out.size() / 2];
    }

    std::vector<double> samples;
    samples.reserve(repeats);
    for (int i = 0; i < repeats; ++i) {
        const auto start = Clock::now();
        auto out = algo(arr, keys);
        const auto end = Clock::now();
        sink = sink + out[out.size() / 2];
        const std::chrono::duration<double, std::micro> elapsed = end - start;
        samples.push_back(elapsed.count());
    }
    std::sort(samples.begin(), samples.end());
    if (sink == std::numeric_limits<i64>::min()) {
        std::cerr << sink;
    }
    return samples[samples.size() / 2];
}

int main()
{
    const std::vector<i64> ns = {1'000'000, 10'000'000};
    const std::vector<i64> qs = {1'000, 100'000};
    const std::vector<std::string> shapes = {
        "dense", "medium", "sparse", "clustered", "mostly_monotonic", "random"
    };

    std::cout << "SHAPE_SCREEN\n";
    std::cout << "N,Q,shape,current_us,galloping_us,hybrid_us,"
                 "gallop_vs_current,hybrid_vs_current\n";

    for (const i64 n : ns) {
        std::vector<std::int32_t> arr(n);
        for (i64 i = 0; i < n; ++i) {
            arr[i] = static_cast<std::int32_t>(i);
        }

        for (const i64 q : qs) {
            for (const auto& shape : shapes) {
                auto keys = make_queries(n, q, shape, 42);
                if (keys.empty()) {
                    continue;
                }

                const auto a = current_batched(arr, keys);
                const auto c = galloping(arr, keys);
                const auto h = hybrid_sorted_gate(arr, keys);
                if (!verify(arr, keys, a) || !verify(arr, keys, c) || !verify(arr, keys, h)) {
                    return 2;
                }

                const double ta = time_algo(current_batched, arr, keys);
                const double tc = time_algo(galloping, arr, keys);
                const double th = time_algo(hybrid_sorted_gate, arr, keys);

                std::cout << n << ',' << q << ',' << shape << ','
                          << std::fixed << std::setprecision(3)
                          << ta << ',' << tc << ',' << th << ','
                          << (tc / ta) << ',' << (th / ta) << '\n';
            }
        }
    }

    std::cout << "DELTA_SWEEP\n";
    std::cout << "N,Q,delta,current_us,galloping_us,hybrid_us,"
                 "gallop_vs_current,hybrid_vs_current\n";

    const std::vector<std::pair<i64, std::vector<i64>>> sweep_cases = {
        {1'000, {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1'024, 2'048, 4'096, 8'192}},
        {100'000, {1, 2, 4, 8, 16, 32, 64}},
    };

    const i64 n = 10'000'000;
    std::vector<std::int32_t> arr(n);
    for (i64 i = 0; i < n; ++i) {
        arr[i] = static_cast<std::int32_t>(i);
    }

    for (const auto& [q, deltas] : sweep_cases) {
        for (const i64 delta : deltas) {
            auto keys = make_fixed_delta(n, q, delta);
            if (keys.empty()) {
                continue;
            }

            const auto a = current_batched(arr, keys);
            const auto c = galloping(arr, keys);
            const auto h = hybrid_sorted_gate(arr, keys);
            if (!verify(arr, keys, a) || !verify(arr, keys, c) || !verify(arr, keys, h)) {
                return 3;
            }

            const double ta = time_algo(current_batched, arr, keys);
            const double tc = time_algo(galloping, arr, keys);
            const double th = time_algo(hybrid_sorted_gate, arr, keys);

            std::cout << n << ',' << q << ',' << delta << ','
                      << std::fixed << std::setprecision(3)
                      << ta << ',' << tc << ',' << th << ','
                      << (tc / ta) << ',' << (th / ta) << '\n';
        }
    }

    return 0;
}
