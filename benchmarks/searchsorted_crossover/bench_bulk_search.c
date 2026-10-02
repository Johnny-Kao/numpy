#include <stddef.h>
#include <stdint.h>

void binary_search_left_i64(
    const int64_t *a, size_t n,
    const int64_t *q, size_t m,
    int64_t *out)
{
    for (size_t j = 0; j < m; ++j) {
        size_t lo = 0;
        size_t hi = n;
        const int64_t x = q[j];
        while (lo < hi) {
            const size_t mid = lo + (hi - lo) / 2;
            if (a[mid] < x) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        out[j] = (int64_t)lo;
    }
}

void merge_search_left_i64(
    const int64_t *a, size_t n,
    const int64_t *q, size_t m,
    int64_t *out)
{
    size_t i = 0;
    for (size_t j = 0; j < m; ++j) {
        const int64_t x = q[j];
        while (i < n && a[i] < x) {
            ++i;
        }
        out[j] = (int64_t)i;
    }
}
