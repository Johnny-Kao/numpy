/* -*- c -*- */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"

#include "npy_binsearch.h"
#include "npy_sort.h"
#include "numpy_tag.hpp"

#include <array>
#include <cstdlib>
#include <cstring>
#include <functional>  // for std::less and std::less_equal

// Enumerators for the variant of binsearch
enum arg_t
{
    noarg,
    arg
};
enum side_t
{
    left,
    right
};

// Mapping from enumerators to comparators
template <class Tag, side_t side>
struct side_to_cmp;

template <class Tag>
struct side_to_cmp<Tag, left> {
    static constexpr auto value = Tag::less;
};

template <class Tag>
struct side_to_cmp<Tag, right> {
    static constexpr auto value = Tag::less_equal;
};

template <side_t side>
struct side_to_generic_cmp;

template <>
struct side_to_generic_cmp<left> {
    using type = std::less<int>;
};

template <>
struct side_to_generic_cmp<right> {
    using type = std::less_equal<int>;
};

/*
 *****************************************************************************
 **                            NUMERIC SEARCHES                             **
 *****************************************************************************
 */
template <class Tag, side_t side>
static void
binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
          npy_intp key_len, npy_intp arr_str, npy_intp key_str,
          npy_intp ret_str, PyArrayObject *, PyArrayObject *,
          PyArray_BinSearchCompareFunc *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;
    auto less = Tag::less;

    if (arr_len <= 0) {
        for (npy_intp i = 0; i < key_len; ++i) {
            *(npy_intp *)(ret + i * ret_str) = 0;
        }
        return;
    }
    if (key_len == 0) {
        return;
    }

    /*
     * Research-only selector tournament.
     *
     * The mode is read once per dtype/side specialization. The benchmark
     * correctness warm-up initializes it before timing, so getenv/strcmp are
     * not part of the timed search loop.
     *
     * This branch is intentionally not an upstream/production patch. It lets
     * one wheel exercise several execution policies so GitHub Actions can
     * amortize checkout/build/startup cost across a broad experiment matrix.
     */
    static const int research_mode = []() {
        const char *mode = std::getenv("NPY_SEARCHSORTED_RESEARCH_MODE");
        if (mode == nullptr || std::strcmp(mode, "current") == 0) {
            return 0;
        }
        if (std::strcmp(mode, "prev_bound") == 0) {
            return 1;
        }
        if (std::strcmp(mode, "galloping") == 0) {
            return 2;
        }
        if (std::strcmp(mode, "hybrid_full") == 0) {
            return 3;
        }
        if (std::strcmp(mode, "hybrid_sample16") == 0) {
            return 4;
        }
        if (std::strcmp(mode, "hybrid_sample32") == 0) {
            return 5;
        }
        if (std::strcmp(mode, "hybrid_chunk") == 0) {
            return 6;
        }
        if (std::strcmp(mode, "current_chunk") == 0) {
            return 7;
        }
        if (std::strcmp(mode, "gallop_chunk") == 0) {
            return 8;
        }
        if (std::strcmp(mode, "adaptive") == 0) {
            return 9;
        }
        if (std::strcmp(mode, "adaptive_dynamic") == 0) {
            return 10;
        }
        return 0;
    }();

    static const npy_intp research_chunk_size = []() {
        const char *raw = std::getenv("NPY_SEARCHSORTED_RESEARCH_CHUNK");
        if (raw == nullptr) {
            return (npy_intp)16;
        }
        const long value = std::strtol(raw, nullptr, 10);
        if (value < 1 || value > 4096) {
            return (npy_intp)16;
        }
        return (npy_intp)value;
    }();

    static const int research_policy = []() {
        const char *raw = std::getenv("NPY_SEARCHSORTED_RESEARCH_POLICY");
        if (raw == nullptr) {
            return 0;
        }
        const long value = std::strtol(raw, nullptr, 10);
        return (value >= 0 && value <= 7) ? (int)value : 0;
    }();

    auto env_intp = [](const char *name, npy_intp fallback,
                       npy_intp min_value, npy_intp max_value) {
        const char *raw = std::getenv(name);
        if (raw == nullptr) {
            return fallback;
        }
        const long value = std::strtol(raw, nullptr, 10);
        if (value < min_value || value > max_value) {
            return fallback;
        }
        return (npy_intp)value;
    };

    auto run_current = [&](const char *keys, char *rets, npy_intp count) {
        if (count == 0) {
            return;
        }

        npy_intp interval_length = arr_len;
        npy_intp half = interval_length >> 1;
        interval_length -= half;

        const T mid_val = *(const T *)(arr + half * arr_str);
        for (npy_intp i = 0; i < count; ++i) {
            const T key_val = *(const T *)(keys + i * key_str);
            *(npy_intp *)(rets + i * ret_str) = cmp(mid_val, key_val) * half;
        }

        while (interval_length > 1) {
            half = interval_length >> 1;
            interval_length -= half;
            for (npy_intp i = 0; i < count; ++i) {
                npy_intp &base = *(npy_intp *)(rets + i * ret_str);
                const T pivot = *(const T *)(arr + (base + half) * arr_str);
                const T key_val = *(const T *)(keys + i * key_str);
                base += cmp(pivot, key_val) * half;
            }
        }

        for (npy_intp i = 0; i < count; ++i) {
            npy_intp &base = *(npy_intp *)(rets + i * ret_str);
            const T key_val = *(const T *)(keys + i * key_str);
            base += cmp(*(const T *)(arr + base * arr_str), key_val);
        }
    };

    auto run_prev_bound = [&](const char *keys, char *rets, npy_intp count) {
        if (count == 0) {
            return;
        }

        npy_intp min_idx = 0;
        npy_intp max_idx = arr_len;
        T last_key_val = *(const T *)keys;

        for (npy_intp i = 0; i < count; ++i) {
            const T key_val = *(const T *)(keys + i * key_str);
            if (cmp(last_key_val, key_val)) {
                max_idx = arr_len;
            }
            else {
                min_idx = 0;
                max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
            }
            last_key_val = key_val;

            while (min_idx < max_idx) {
                const npy_intp mid_idx =
                        min_idx + ((max_idx - min_idx) >> 1);
                const T pivot = *(const T *)(arr + mid_idx * arr_str);
                if (cmp(pivot, key_val)) {
                    min_idx = mid_idx + 1;
                }
                else {
                    max_idx = mid_idx;
                }
            }
            *(npy_intp *)(rets + i * ret_str) = min_idx;
        }
    };

    auto run_galloping = [&](const char *keys, char *rets, npy_intp count) {
        if (count == 0) {
            return;
        }

        T last_key_val = *(const T *)keys;
        npy_intp previous_pos = 0;

        for (npy_intp i = 0; i < count; ++i) {
            const T key_val = *(const T *)(keys + i * key_str);
            npy_intp min_idx = 0;
            npy_intp max_idx = arr_len;

            if (i > 0 && !less(key_val, last_key_val)) {
                min_idx = previous_pos;
                if (min_idx < arr_len &&
                        cmp(*(const T *)(arr + min_idx * arr_str), key_val)) {
                    const npy_intp origin = min_idx;
                    npy_intp step = 1;
                    min_idx = origin + 1;

                    while (true) {
                        const npy_intp remaining = arr_len - 1 - origin;
                        if (step > remaining) {
                            max_idx = arr_len;
                            break;
                        }
                        const npy_intp probe = origin + step;
                        const T probe_val =
                                *(const T *)(arr + probe * arr_str);
                        if (!cmp(probe_val, key_val)) {
                            max_idx = probe + 1;
                            break;
                        }
                        min_idx = probe + 1;
                        if (step > (remaining >> 1)) {
                            max_idx = arr_len;
                            break;
                        }
                        step <<= 1;
                    }
                }
                else {
                    max_idx =
                            (min_idx < arr_len) ? (min_idx + 1) : arr_len;
                }
            }

            while (min_idx < max_idx) {
                const npy_intp mid_idx =
                        min_idx + ((max_idx - min_idx) >> 1);
                const T pivot = *(const T *)(arr + mid_idx * arr_str);
                if (cmp(pivot, key_val)) {
                    min_idx = mid_idx + 1;
                }
                else {
                    max_idx = mid_idx;
                }
            }

            *(npy_intp *)(rets + i * ret_str) = min_idx;
            previous_pos = min_idx;
            last_key_val = key_val;
        }
    };

    auto fully_nondecreasing =
            [&](const char *keys, npy_intp count) -> bool {
        for (npy_intp i = 1; i < count; ++i) {
            const T prev = *(const T *)(keys + (i - 1) * key_str);
            const T cur = *(const T *)(keys + i * key_str);
            if (less(cur, prev)) {
                return false;
            }
        }
        return true;
    };

    auto sampled_nondecreasing =
            [&](const char *keys, npy_intp count, npy_intp samples) -> bool {
        if (count < 2) {
            return true;
        }
        const npy_intp checks = (count - 1 < samples) ? count - 1 : samples;
        for (npy_intp j = 1; j <= checks; ++j) {
            npy_intp i = (j * (count - 1)) / checks;
            if (i < 1) {
                i = 1;
            }
            const T prev = *(const T *)(keys + (i - 1) * key_str);
            const T cur = *(const T *)(keys + i * key_str);
            if (less(cur, prev)) {
                return false;
            }
        }
        return true;
    };

    auto sampled_inversions =
            [&](const char *keys, npy_intp count, npy_intp samples,
                npy_intp stop_after) -> npy_intp {
        if (count < 2 || samples <= 0) {
            return 0;
        }
        const npy_intp checks = (count - 1 < samples) ? count - 1 : samples;
        npy_intp inversions = 0;
        for (npy_intp j = 1; j <= checks; ++j) {
            npy_intp i = (j * (count - 1)) / checks;
            if (i < 1) {
                i = 1;
            }
            const T prev = *(const T *)(keys + (i - 1) * key_str);
            const T cur = *(const T *)(keys + i * key_str);
            if (less(cur, prev)) {
                ++inversions;
                if (inversions > stop_after) {
                    break;
                }
            }
        }
        return inversions;
    };

    struct adaptive_policy_t {
        npy_intp activate_q;
        npy_intp stage2_q;
        npy_intp stage3_q;
        npy_intp samples1;
        npy_intp samples2;
        npy_intp samples3;
        npy_intp inv1;
        npy_intp inv2;
        npy_intp inv3;
    };

    static constexpr adaptive_policy_t adaptive_policies[] = {
        /* P0 conservative */
        {64, 1024, 16384, 4, 8, 16, 0, 0, 0},
        /* P1 balanced strict */
        {32, 512, 8192, 4, 8, 16, 0, 0, 0},
        /* P2 balanced tolerant */
        {32, 512, 8192, 4, 8, 24, 0, 1, 2},
        /* P3 aggressive */
        {16, 256, 4096, 4, 8, 32, 0, 1, 3},
        /* P4 heavy-only, deeper validation */
        {128, 2048, 32768, 8, 16, 64, 0, 1, 3},
        /* P5 early activation, strict */
        {16, 128, 2048, 4, 8, 16, 0, 0, 0},
        /* P6 tolerant large-workload */
        {64, 1024, 8192, 4, 12, 32, 0, 1, 2},
        /* P7 ultra-conservative random protection */
        {256, 4096, 65536, 8, 16, 32, 0, 0, 1},
    };

    if (research_mode == 0) {
        run_current(key, ret, key_len);
        return;
    }
    if (research_mode == 1) {
        run_prev_bound(key, ret, key_len);
        return;
    }
    if (research_mode == 2) {
        run_galloping(key, ret, key_len);
        return;
    }
    if (research_mode == 3) {
        if (fully_nondecreasing(key, key_len)) {
            run_galloping(key, ret, key_len);
        }
        else {
            run_current(key, ret, key_len);
        }
        return;
    }
    if (research_mode == 4 || research_mode == 5) {
        const npy_intp samples = (research_mode == 4) ? 16 : 32;
        if (sampled_nondecreasing(key, key_len, samples)) {
            run_galloping(key, ret, key_len);
        }
        else {
            run_current(key, ret, key_len);
        }
        return;
    }

    if (research_mode == 9) {
        const adaptive_policy_t &policy =
                adaptive_policies[research_policy];

        if (key_len < policy.activate_q) {
            run_current(key, ret, key_len);
            return;
        }

        npy_intp samples = policy.samples1;
        npy_intp allowed = policy.inv1;
        if (key_len >= policy.stage3_q) {
            samples = policy.samples3;
            allowed = policy.inv3;
        }
        else if (key_len >= policy.stage2_q) {
            samples = policy.samples2;
            allowed = policy.inv2;
        }

        /*
         * The selector cost is capped by the policy's sample budget.
         * Stop as soon as the inversion allowance is exceeded so random
         * inputs pay only a small fraction of the maximum inspection cost.
         */
        const npy_intp inversions =
                sampled_inversions(key, key_len, samples, allowed);
        if (inversions <= allowed) {
            run_galloping(key, ret, key_len);
        }
        else {
            run_current(key, ret, key_len);
        }
        return;
    }

    if (research_mode == 10) {
        adaptive_policy_t policy = {
            env_intp("NPY_SS_ACTIVATE_Q", 128, 1, 100000000),
            env_intp("NPY_SS_STAGE2_Q", 2048, 1, 100000000),
            env_intp("NPY_SS_STAGE3_Q", 32768, 1, 100000000),
            env_intp("NPY_SS_SAMPLES1", 8, 1, 256),
            env_intp("NPY_SS_SAMPLES2", 16, 1, 256),
            env_intp("NPY_SS_SAMPLES3", 64, 1, 512),
            env_intp("NPY_SS_INV1", 0, 0, 64),
            env_intp("NPY_SS_INV2", 1, 0, 64),
            env_intp("NPY_SS_INV3", 3, 0, 128),
        };

        if (key_len < policy.activate_q) {
            run_current(key, ret, key_len);
            return;
        }

        npy_intp samples = policy.samples1;
        npy_intp allowed = policy.inv1;
        if (key_len >= policy.stage3_q) {
            samples = policy.samples3;
            allowed = policy.inv3;
        }
        else if (key_len >= policy.stage2_q) {
            samples = policy.samples2;
            allowed = policy.inv2;
        }

        const npy_intp inversions =
                sampled_inversions(key, key_len, samples, allowed);
        if (inversions <= allowed) {
            run_galloping(key, ret, key_len);
        }
        else {
            run_current(key, ret, key_len);
        }
        return;
    }

    const npy_intp chunk_size = research_chunk_size;
    for (npy_intp offset = 0; offset < key_len; offset += chunk_size) {
        const npy_intp remaining = key_len - offset;
        const npy_intp count =
                (remaining < chunk_size) ? remaining : chunk_size;
        const char *chunk_keys = key + offset * key_str;
        char *chunk_rets = ret + offset * ret_str;

        if (research_mode == 7) {
            run_current(chunk_keys, chunk_rets, count);
        }
        else if (research_mode == 8) {
            run_galloping(chunk_keys, chunk_rets, count);
        }
        else if (fully_nondecreasing(chunk_keys, count)) {
            run_galloping(chunk_keys, chunk_rets, count);
        }
        else {
            run_current(chunk_keys, chunk_rets, count);
        }
    }
}

template <class Tag, side_t side>
static int
argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
             npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
             npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
             PyArrayObject *, PyArrayObject *, PyArray_BinSearchCompareFunc *)
{
    using T = typename Tag::type;
    auto cmp = side_to_cmp<Tag, side>::value;

    // If the array length is 0 we return all 0s
    if (arr_len <= 0) {
        for (npy_intp i = 0; i < key_len; ++i) {
            *(npy_intp *)(ret + i * ret_str) = 0;
        }
        return 0;
    }

    npy_intp interval_length = arr_len;
    npy_intp half = interval_length >> 1;
    interval_length -= half; // length -> ceil(length / 2)

    npy_intp base = 0;
    npy_intp mid_idx = *(npy_intp *)(sort + (base + half) * sort_str);
    if (mid_idx < 0 || mid_idx >= arr_len) {
        return -1;
    }
    const T mid_val = *(const T *)(arr + mid_idx * arr_str);

    for (npy_intp i = 0; i < key_len; ++i) {
        const T key_val = *(const T *)(key + i * key_str);
        *(npy_intp *)(ret + i * ret_str) = cmp(mid_val, key_val) * half;
    }

    while (interval_length > 1) {
        npy_intp half = interval_length >> 1;
        interval_length -= half; // length -> ceil(length / 2)

        for (npy_intp i = 0; i < key_len; ++i) {
            npy_intp &base = *(npy_intp *)(ret + i * ret_str);
            npy_intp mid_idx = *(npy_intp *)(sort + (base + half) * sort_str);
            if (mid_idx < 0 || mid_idx >= arr_len) {
                return -1;
            }
            const T mid_val = *(const T *)(arr + mid_idx * arr_str);
            const T key_val = *(const T *)(key + i * key_str);
            base += cmp(mid_val, key_val) * half;
        }
    }

    for (npy_intp i = 0; i < key_len; ++i) {
        npy_intp &base = *(npy_intp *)(ret + i * ret_str);
        npy_intp mid_idx = *(npy_intp *)(sort + base * sort_str);
        if (mid_idx < 0 || mid_idx >= arr_len) {
            return -1;
        }
        const T key_val = *(const T *)(key + i * key_str);
        base += cmp(*(const T *)(arr + mid_idx * arr_str), key_val);
    }
    return 0;
}

/*
 *****************************************************************************
 **                             GENERIC SEARCH                              **
 *****************************************************************************
 */

template <side_t side>
static void
npy_binsearch(const char *arr, const char *key, char *ret, npy_intp arr_len,
              npy_intp key_len, npy_intp arr_str, npy_intp key_str,
              npy_intp ret_str, PyArrayObject *key_arr, PyArrayObject *arr_arr,
              PyArray_BinSearchCompareFunc *compare)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        /* last_key and key are both elements of the key array */
        if (Cmp{}(compare(last_key, key, key_arr, key_arr), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const char *arr_ptr = arr + mid_idx * arr_str;

            /* arr_ptr belongs to the haystack, key to the key array */
            if (Cmp{}(compare(arr_ptr, key, arr_arr, key_arr), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
}

template <side_t side>
static int
npy_argbinsearch(const char *arr, const char *key, const char *sort, char *ret,
                 npy_intp arr_len, npy_intp key_len, npy_intp arr_str,
                 npy_intp key_str, npy_intp sort_str, npy_intp ret_str,
                 PyArrayObject *key_arr, PyArrayObject *arr_arr,
                 PyArray_BinSearchCompareFunc *compare)
{
    using Cmp = typename side_to_generic_cmp<side>::type;
    npy_intp min_idx = 0;
    npy_intp max_idx = arr_len;
    const char *last_key = key;

    for (; key_len > 0; key_len--, key += key_str, ret += ret_str) {
        /*
         * Updating only one of the indices based on the previous key
         * gives the search a big boost when keys are sorted, but slightly
         * slows down things for purely random ones.
         */
        /* last_key and key are both elements of the key array */
        if (Cmp{}(compare(last_key, key, key_arr, key_arr), 0)) {
            max_idx = arr_len;
        }
        else {
            min_idx = 0;
            max_idx = (max_idx < arr_len) ? (max_idx + 1) : arr_len;
        }

        last_key = key;

        while (min_idx < max_idx) {
            const npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);
            const npy_intp sort_idx = *(npy_intp *)(sort + mid_idx * sort_str);
            const char *arr_ptr;

            if (sort_idx < 0 || sort_idx >= arr_len) {
                return -1;
            }

            arr_ptr = arr + sort_idx * arr_str;

            /* arr_ptr belongs to the haystack, key to the key array */
            if (Cmp{}(compare(arr_ptr, key, arr_arr, key_arr), 0)) {
                min_idx = mid_idx + 1;
            }
            else {
                max_idx = mid_idx;
            }
        }
        *(npy_intp *)ret = min_idx;
    }
    return 0;
}

/*
 *****************************************************************************
 **                             GENERATOR                                   **
 *****************************************************************************
 */

template <arg_t arg>
struct binsearch_base;

template <>
struct binsearch_base<arg> {
    using function_type = PyArray_ArgBinSearchFunc *;
    struct value_type {
        int typenum;
        function_type binsearch[NPY_NSEARCHSIDES];
    };
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&argbinsearch<Tags, left>,
                            (function_type)argbinsearch<Tags, right>}}...};
    }
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_argbinsearch<left>,
            (function_type)&npy_argbinsearch<right>};
};
constexpr std::array<binsearch_base<arg>::function_type, 2>
        binsearch_base<arg>::npy_map;

template <>
struct binsearch_base<noarg> {
    using function_type = PyArray_BinSearchFunc *;
    struct value_type {
        int typenum;
        function_type binsearch[NPY_NSEARCHSIDES];
    };
    template <class... Tags>
    static constexpr std::array<value_type, sizeof...(Tags)>
    make_binsearch_map(npy::taglist<Tags...>)
    {
        return std::array<value_type, sizeof...(Tags)>{
                value_type{Tags::type_value,
                           {(function_type)&binsearch<Tags, left>,
                            (function_type)binsearch<Tags, right>}}...};
    }
    static constexpr std::array<function_type, 2> npy_map = {
            (function_type)&npy_binsearch<left>,
            (function_type)&npy_binsearch<right>};
};
constexpr std::array<binsearch_base<noarg>::function_type, 2>
        binsearch_base<noarg>::npy_map;

// Handle generation of all binsearch variants
template <arg_t arg>
struct binsearch_t : binsearch_base<arg> {
    using binsearch_base<arg>::make_binsearch_map;
    using value_type = typename binsearch_base<arg>::value_type;

    using taglist = npy::taglist<
            /* If adding new types, make sure to keep them ordered by type num
             */
            npy::bool_tag, npy::byte_tag, npy::ubyte_tag, npy::short_tag,
            npy::ushort_tag, npy::int_tag, npy::uint_tag, npy::long_tag,
            npy::ulong_tag, npy::longlong_tag, npy::ulonglong_tag,
            npy::float_tag, npy::double_tag, npy::longdouble_tag, 
            npy::cfloat_tag, npy::cdouble_tag, npy::clongdouble_tag, 
            npy::datetime_tag, npy::timedelta_tag, npy::half_tag>;

    static constexpr std::array<value_type, taglist::size> map =
            make_binsearch_map(taglist());
};

template <arg_t arg>
constexpr std::array<typename binsearch_t<arg>::value_type,
                     binsearch_t<arg>::taglist::size>
        binsearch_t<arg>::map;

template <arg_t arg>
static inline typename binsearch_t<arg>::function_type
_get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    using binsearch = binsearch_t<arg>;
    npy_intp nfuncs = binsearch::map.size();
    npy_intp min_idx = 0;
    npy_intp max_idx = nfuncs;
    int type = dtype->type_num;

    if ((int)side >= (int)NPY_NSEARCHSIDES) {
        return NULL;
    }

    /*
     * It seems only fair that a binary search function be searched for
     * using a binary search...
     */
    while (min_idx < max_idx) {
        npy_intp mid_idx = min_idx + ((max_idx - min_idx) >> 1);

        if (binsearch::map[mid_idx].typenum < type) {
            min_idx = mid_idx + 1;
        }
        else {
            max_idx = mid_idx;
        }
    }

    if (min_idx < nfuncs && binsearch::map[min_idx].typenum == type) {
        return binsearch::map[min_idx].binsearch[side];
    }

    if (PyDataType_GetArrFuncs(dtype)->compare) {
        return binsearch::npy_map[side];
    }

    return NULL;
}

/*
 *****************************************************************************
 **                            C INTERFACE                                  **
 *****************************************************************************
 */
extern "C" {
NPY_NO_EXPORT PyArray_BinSearchFunc *
get_binsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return _get_binsearch_func<noarg>(dtype, side);
}

NPY_NO_EXPORT PyArray_ArgBinSearchFunc *
get_argbinsearch_func(PyArray_Descr *dtype, NPY_SEARCHSIDE side)
{
    return _get_binsearch_func<arg>(dtype, side);
}
}
