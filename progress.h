#pragma once

// progress dumping faciltiies

#include <cstdint>
#include <cstdio>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "util.h"

// frame range (inclusive indexes)
struct FrameRange {
    uint32_t low = 0;
    uint32_t high = 0;
};
struct ChunkData {
    uint32_t idx = 0;
    uint32_t low = 0;
    uint32_t high = 0;
};

// DiViEn progress file format

// Format goes like this:
// <chunk_idx> <low_frame> <high_frame>
// The chunks are allowed to be stored in any order.
// They will usually be close to sorted but not sorted.
// Each number is a 4 byte, unsigned int stored in Little endian.
// The progress file is invalid if any frame ranges overlap, or if
// there are duplicate chunk_idx, or if the number of bytes is
// not a multiple of 12.
// The [low_frame, high_frame] is an inclusive bound, represented like that.
// The progress file may be empty.

// To view the file:
// od -t u4 <progress_file.dvn>

// TODO unordered_map replace with something more minimal
using chunk_hmap = std::unordered_map<uint32_t, FrameRange>;

// TODO; detect invalid/incompatible block size

// TODO: error handling
// Intel TBB?
inline chunk_hmap parse_chunks(const char* pfile) {
    FILE* fptr = fopen(pfile, "rb");

    chunk_hmap map;

    uint32_t buffer[3];
    while (fread((uint32_t*)buffer, sizeof(uint32_t), 3, fptr) == 3) {
        // TODO calculate frame sum here as well
        // or somewhere or whatever.
        DvAssert(buffer[2] >= buffer[1]);
        map.insert(
            {buffer[0], FrameRange{.low = buffer[1], .high = buffer[2]}});
    }

    DvAssert(fclose(fptr) == 0);

    return map;
}

// frame indexes are inclusive.
inline void dump_chunk(std::mutex& f_mutex, chunk_hmap& finished_chunks,
                       const char* pfile, ChunkData cd) {
    std::lock_guard<std::mutex> lock(f_mutex);
    // ---- we have lock on mutex now

    finished_chunks.insert(
        {cd.idx, FrameRange{.low = cd.low, .high = cd.high}});

    FILE* fptr = fopen(pfile, "wb");
    DvAssert(fptr);

    // need to write whole file again so that we don't have the problem
    // of user ctrl+Cing and file handle is lost abruptly.
    // I guess there's a slim chance user can corrupt while we are writing
    // progress file but I doubt that will happen very often at all.
    // TODO maybe research if keeping open file handle is actually better.
    for (const auto [idx, chunk] : finished_chunks) {
        DvAssert(fwrite(&idx, 4, 1, fptr) == 1);
        DvAssert(fwrite(&chunk.low, 4, 1, fptr) == 1);
        DvAssert(fwrite(&chunk.high, 4, 1, fptr) == 1);
    }

    DvAssert(fclose(fptr) == 0);
}
