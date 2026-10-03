/* -*- c -*- */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"

#include "npy_binsearch.h"
#include "npy_sort.h"
#include "numpy_tag.hpp"

#include <array>
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

    // If the array length is 0 we return all 0s
    if (arr_len <= 0) {
        for (npy_intp i = 0; i < key_len; ++i) {
            *(npy_intp *)(ret + i * ret_str) = 0;
        }
        return;
    }

    /*
     * Small batches retain the current batched binary-search path unchanged.
     * For larger batches, the first few batched levels already produce a
     * coarse insertion-position trajectory in ret.  A tiny fixed sample of
     * that state is enough to identify strong locality without scanning the
     * query values or allocating additional storage.
     */
    /*
     * Keep the initial activation point conservative across CPUs. The
     * crossover was lower on the tested AMD runners than on Intel, so this is
     * a portable default rather than a claim that 2048 is globally optimal.
     */
    constexpr npy_intp locality_min_keys = 2048;
    constexpr npy_intp locality_samples = 16;
    constexpr npy_intp locality_levels = 3;

    if (key_len < locality_min_keys) {
        npy_intp interval_length = arr_len;
        npy_intp half = interval_length >> 1;
        interval_length -= half; // length -> ceil(length / 2)

        const T mid_val = *(const T *)(arr + half * arr_str);

        for (npy_intp i = 0; i < key_len; ++i) {
            const T key_val = *(const T *)(key + i * key_str);
            *(npy_intp *)(ret + i * ret_str) = cmp(mid_val, key_val) * half;
        }

        while (interval_length > 1) {
            half = interval_length >> 1;
            interval_length -= half; // length -> ceil(length / 2)

            for (npy_intp i = 0; i < key_len; ++i) {
                npy_intp &base = *(npy_intp *)(ret + i * ret_str);
                const T pivot = *(const T *)(arr + (base + half) * arr_str);
                const T key_val = *(const T *)(key + i * key_str);
                base += cmp(pivot, key_val) * half;
            }
        }

        for (npy_intp i = 0; i < key_len; ++i) {
            npy_intp &base = *(npy_intp *)(ret + i * ret_str);
            const T key_val = *(const T *)(key + i * key_str);
            base += cmp(*(const T *)(arr + base * arr_str), key_val);
        }
        return;
    }

    npy_intp interval_length = arr_len;
    npy_intp half = interval_length >> 1;
    interval_length -= half; // length -> ceil(length / 2)

    const T mid_val = *(const T *)(arr + half * arr_str);

    for (npy_intp i = 0; i < key_len; ++i) {
        const T key_val = *(const T *)(key + i * key_str);
        *(npy_intp *)(ret + i * ret_str) = cmp(mid_val, key_val) * half;
    }

    npy_intp completed_levels = 1;
    while (interval_length > 1 &&
           completed_levels < locality_levels) {
        half = interval_length >> 1;
        interval_length -= half; // length -> ceil(length / 2)

        for (npy_intp i = 0; i < key_len; ++i) {
            npy_intp &base = *(npy_intp *)(ret + i * ret_str);
            const T pivot = *(const T *)(arr + (base + half) * arr_str);
            const T key_val = *(const T *)(key + i * key_str);
            base += cmp(pivot, key_val) * half;
        }
        ++completed_levels;
    }

    bool use_local_finish = false;
    if (interval_length > 1) {
        const npy_intp last = key_len - 1;
        npy_intp prev = *(npy_intp *)ret;
        int direction = 0;
        use_local_finish = true;

        for (npy_intp j = 1; j <= locality_samples; ++j) {
            const npy_intp i = (j * last) >> 4;
            const npy_intp pos = *(npy_intp *)(ret + i * ret_str);

            if (pos > prev) {
                if (direction < 0) {
                    use_local_finish = false;
                    break;
                }
                direction = 1;
            }
            else if (pos < prev) {
                if (direction > 0) {
                    use_local_finish = false;
                    break;
                }
                direction = -1;
            }
            prev = pos;
        }
    }

    if (!use_local_finish) {
        while (interval_length > 1) {
            half = interval_length >> 1;
            interval_length -= half; // length -> ceil(length / 2)

            for (npy_intp i = 0; i < key_len; ++i) {
                npy_intp &base = *(npy_intp *)(ret + i * ret_str);
                const T pivot = *(const T *)(arr + (base + half) * arr_str);
                const T key_val = *(const T *)(key + i * key_str);
                base += cmp(pivot, key_val) * half;
            }
        }

        for (npy_intp i = 0; i < key_len; ++i) {
            npy_intp &base = *(npy_intp *)(ret + i * ret_str);
            const T key_val = *(const T *)(key + i * key_str);
            base += cmp(*(const T *)(arr + base * arr_str), key_val);
        }
        return;
    }

    /*
     * Finish from the already-computed coarse bounds.  The previous exact
     * insertion position is a valid lower bound for nondecreasing keys and a
     * valid upper bound for decreasing keys.  Exponential probing narrows that
     * bound before the final binary search.  The per-key coarse interval
     * remains the correctness envelope even when local order briefly changes.
     */
    T previous_key = *(const T *)key;
    npy_intp previous_pos = 0;

    for (npy_intp i = 0; i < key_len; ++i) {
        const T key_val = *(const T *)(key + i * key_str);
        const npy_intp coarse_base = *(npy_intp *)(ret + i * ret_str);
        npy_intp min_idx = coarse_base;
        npy_intp max_idx = coarse_base + interval_length;
        if (max_idx > arr_len) {
            max_idx = arr_len;
        }

        if (i > 0) {
            if (!less(key_val, previous_key)) {
                if (previous_pos > min_idx) {
                    min_idx = previous_pos;
                }
                if (min_idx < max_idx &&
                    cmp(*(const T *)(arr + min_idx * arr_str), key_val)) {
                    const npy_intp origin = min_idx;
                    npy_intp step = 1;
                    min_idx = origin + 1;

                    while (origin + step < max_idx) {
                        const npy_intp probe = origin + step;
                        if (!cmp(*(const T *)(arr + probe * arr_str),
                                 key_val)) {
                            max_idx = probe + 1;
                            break;
                        }
                        min_idx = probe + 1;
                        const npy_intp remaining = max_idx - 1 - origin;
                        if (step > (remaining >> 1)) {
                            break;
                        }
                        step <<= 1;
                    }
                }
                else if (min_idx < max_idx) {
                    max_idx = min_idx + 1;
                }
            }
            else {
                if (previous_pos < max_idx) {
                    max_idx = previous_pos;
                }

                if (min_idx < max_idx) {
                    const npy_intp origin = max_idx;
                    npy_intp step = 1;

                    while (origin > min_idx) {
                        const npy_intp distance = origin - min_idx;
                        const npy_intp probe =
                                (step >= distance) ? min_idx : origin - step;

                        if (cmp(*(const T *)(arr + probe * arr_str),
                                key_val)) {
                            min_idx = probe + 1;
                            break;
                        }

                        max_idx = probe;
                        if (probe == min_idx || step > (distance >> 1)) {
                            break;
                        }
                        step <<= 1;
                    }
                }
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

        *(npy_intp *)(ret + i * ret_str) = min_idx;
        previous_pos = min_idx;
        previous_key = key_val;
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
