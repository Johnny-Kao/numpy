#!/usr/bin/env python3
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument("mode", choices=[
    "full_adjacent",
    "right_only",
    "left_only",
    "half_adjacent",
    "alternating",
])
args = p.parse_args()
# Research-only tuning variants for PR #32895.

path = Path("numpy/_core/src/npysort/binsearch.cpp")
text = path.read_text()

needle = """    if (!reversed && direction >= 0 && interval_length > 1) {
"""

if args.mode == "full_adjacent":
    guard = r'''
    if (!reversed && direction >= 0) {
        for (npy_intp j = 0; j <= LOCALITY_SAMPLES && !reversed; ++j) {
            const npy_intp i = (j * last) >> 4;
            const T key_val = *(const T *)(key + i * key_str);
            if (i > 0) {
                const T prev_key_val = *(const T *)(key + (i - 1) * key_str);
                if (less(key_val, prev_key_val)) reversed = true;
            }
            if (!reversed && i + 1 < key_len) {
                const T next_key_val = *(const T *)(key + (i + 1) * key_str);
                if (less(next_key_val, key_val)) reversed = true;
            }
        }
    }

'''
elif args.mode == "right_only":
    guard = r'''
    if (!reversed && direction >= 0) {
        for (npy_intp j = 0; j <= LOCALITY_SAMPLES && !reversed; ++j) {
            const npy_intp i = (j * last) >> 4;
            if (i + 1 < key_len) {
                const T key_val = *(const T *)(key + i * key_str);
                const T next_key_val = *(const T *)(key + (i + 1) * key_str);
                if (less(next_key_val, key_val)) reversed = true;
            }
        }
    }

'''
elif args.mode == "left_only":
    guard = r'''
    if (!reversed && direction >= 0) {
        for (npy_intp j = 0; j <= LOCALITY_SAMPLES && !reversed; ++j) {
            const npy_intp i = (j * last) >> 4;
            if (i > 0) {
                const T key_val = *(const T *)(key + i * key_str);
                const T prev_key_val = *(const T *)(key + (i - 1) * key_str);
                if (less(key_val, prev_key_val)) reversed = true;
            }
        }
    }

'''
elif args.mode == "half_adjacent":
    guard = r'''
    if (!reversed && direction >= 0) {
        for (npy_intp j = 0; j <= LOCALITY_SAMPLES && !reversed; j += 2) {
            const npy_intp i = (j * last) >> 4;
            const T key_val = *(const T *)(key + i * key_str);
            if (i > 0) {
                const T prev_key_val = *(const T *)(key + (i - 1) * key_str);
                if (less(key_val, prev_key_val)) reversed = true;
            }
            if (!reversed && i + 1 < key_len) {
                const T next_key_val = *(const T *)(key + (i + 1) * key_str);
                if (less(next_key_val, key_val)) reversed = true;
            }
        }
    }

'''
else:
    guard = r'''
    if (!reversed && direction >= 0) {
        for (npy_intp j = 0; j <= LOCALITY_SAMPLES && !reversed; ++j) {
            const npy_intp i = (j * last) >> 4;
            const T key_val = *(const T *)(key + i * key_str);
            if ((j & 1) == 0) {
                if (i + 1 < key_len) {
                    const T next_key_val = *(const T *)(key + (i + 1) * key_str);
                    if (less(next_key_val, key_val)) reversed = true;
                }
            }
            else if (i > 0) {
                const T prev_key_val = *(const T *)(key + (i - 1) * key_str);
                if (less(key_val, prev_key_val)) reversed = true;
            }
        }
    }

'''

if needle not in text:
    raise SystemExit("selector insertion point not found")

path.write_text(text.replace(needle, guard + needle, 1))
print(f"Applied research selector variant: {args.mode}")
