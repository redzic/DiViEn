#pragma once

#include <cstdint>
#include <cstdio>

// frame range (inclusive indexes)
struct FrameRange {
    uint32_t low = 0;
    uint32_t high = 0;
};

// progress dumping faciltiies

// DiViEn progress file format!
// All numbers are stored in little-endian format.

// frame indexes are inclusive.
inline void dump_chunk(FrameRange frange) {
    auto lower = frange.low;
    auto upper = frange.high;
    // printf("finished chunk: frame range %d-%d\n", lower, upper);
}
