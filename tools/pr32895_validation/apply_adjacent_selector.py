#!/usr/bin/env python3
from pathlib import Path
import sys

path = Path("numpy/_core/src/npysort/binsearch.cpp")
text = path.read_text()

needle = """    if (!reversed && direction >= 0 && interval_length > 1) {
"""
replacement = """    if (!reversed && direction >= 0) {
        /*
         * Research-only selector hardening: inspect the immediate
         * neighborhood around each deterministic anchor using the dtype's
         * existing total-order comparator.  This adds O(1) comparisons and
         * catches sequences whose sparse anchors are monotone while their
         * interiors are hostile.
         */
        for (npy_intp j = 0; j <= LOCALITY_SAMPLES && !reversed; ++j) {
            const npy_intp i = (j * last) >> 4;
            const T key_val = *(const T *)(key + i * key_str);
            if (i > 0) {
                const T prev_key_val = *(const T *)(key + (i - 1) * key_str);
                if (less(key_val, prev_key_val)) {
                    reversed = true;
                    break;
                }
            }
            if (i + 1 < key_len) {
                const T next_key_val = *(const T *)(key + (i + 1) * key_str);
                if (less(next_key_val, key_val)) {
                    reversed = true;
                    break;
                }
            }
        }
    }

    if (!reversed && direction >= 0 && interval_length > 1) {
"""

if needle not in text:
    raise SystemExit("selector insertion point not found")
path.write_text(text.replace(needle, replacement, 1))
print("Applied research adjacent-monotone selector")
