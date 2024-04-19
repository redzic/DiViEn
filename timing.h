#pragma once

#include <chrono>

inline auto dist_ms(auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

// TODO should we maybe pass by reference?
inline auto dist_us(auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
}

inline auto now() { return std::chrono::high_resolution_clock::now(); }
