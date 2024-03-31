#pragma once

#include "resource.h"
#include <cstdint>
#include <span>
#include <vector>

// TODO (IMPORTANT)
// figure out how to move this asio stuff out of here man

// #include <asio.hpp>
// #include <asio/co_spawn.hpp>
// #include <asio/detached.hpp>
// #include <asio/io_context.hpp>
// #include <asio/ip/tcp.hpp>
// #include <asio/signal_set.hpp>
// #include <asio/write.hpp>

// using asio::awaitable;

// using asio::use_awaitable;

// extern "C" {
// #include <libavformat/avformat.h>
// #include <libavutil/timestamp.h>
// }

// ok so...
// segment_end CAN be called in write_trailer,
// but with an additional flag.

// TODO maybe reduce size of this struct.
// How big can these values really be?
struct Timestamp {
    int64_t dts;
    int64_t pts;

    Timestamp(int64_t dts_, int64_t pts_) : dts(dts_), pts(pts_) {}
};

struct SegmentingData {
    // Wait a second. Packet
    // Packet offsets give you the base index for the timestamps.
    // Thus the length will be different.
    // The length of this vector is equal to the number of segments.
    // TODO optimization idea; get this data directly from the segment
    // muxer instead of having to manually count the packets separately.
    // This would require another ffmpeg reinterpret_cast hack though.
    std::span<uint32_t> packet_offsets;
    // timestamps[i] gives you the dts and pts of the ith packet in the
    // ORIGINAL (unsegmented) video stream.
    std::span<Timestamp> timestamps;

    SegmentingData(std::span<uint32_t> pkt_offs_, std::span<Timestamp> ts_)
        : packet_offsets(pkt_offs_), timestamps(ts_) {}
};

// inclusive range
struct ConcatRange {
    unsigned int low;
    unsigned int high;

    ConcatRange(unsigned int low_, unsigned int high_)
        : low(low_), high(high_) {
        DvAssert(low_ < high_);
    }
};

// can either be range or individual segment
struct Segment {
    unsigned int low;
    // high cannot be 0 under normal circumstances,
    // because ranges are inclusive.
    // it cannot end on 0, that would just be low=0,
    // regardless if it's concatted or not.
    // if high==0, we signal that to mean
    unsigned int high;

    // is individual, non concatted segment
    [[nodiscard]] bool is_indiv() const { return high == 0; }
    [[nodiscard]] bool is_range() const { return !is_indiv(); }

    void fmt_name(char* buffer) const {
        if (is_indiv()) [[likely]] {
            (void)sprintf(buffer, "OUTPUT%d.mp4", low);

        } else {
            (void)sprintf(buffer, "OUTPUT_%d_%d.mp4", low, high);
        }
    }

    Segment(unsigned int low_, unsigned int high_) : low(low_), high(high_) {}
};

// Returns 0 for success, <0 for error.
// In theory this could be parallelized a decent bit.
// First perhaps we could separate the reading of packets on the
// input stream and writing of packets on the output stream
// to be on 2 separate threads. But how necessary is that...
// Probably not really much.
// TODO parallelize this code if possible.
[[nodiscard]] int segment_video(const char* in_filename,
                                const char* out_filename,
                                unsigned int& nb_segments,
                                std::vector<Timestamp>& timestamps);

[[nodiscard]] std::vector<ConcatRange>
fix_broken_segments(unsigned int num_segments,
                    std::vector<uint32_t>& packet_offsets,
                    std::span<Timestamp> timestamps);

// 250 frames per segment, 1024 segments
// more accurate estimate is maybe 120-250. Will depend on video of
// course.

// reasonable estimate for initial allocation amount
constexpr size_t EST_NB_SEGMENTS = 1100;

// reasonable estimate for packets per segment
constexpr size_t EST_PKTS_PER_SEG = 140;

// TODO should read up on the "dirty" aspect of macros or whatever
// and what workarounds we should use.
// macro version
#define ITER_SEGFILES(macro_use_file, macro_nb_segments, macro_segs)           \
    {                                                                          \
        std::array<char, 64> fname_buf{};                                      \
        uint32_t i = 0;                                                        \
        for (const auto& r : (macro_segs)) {                                   \
            while (i < r.low) {                                                \
                (void)snprintf(fname_buf.data(), fname_buf.size(),             \
                               "OUTPUT%d.mp4", i++);                           \
                macro_use_file(fname_buf.data());                              \
            }                                                                  \
            (void)snprintf(fname_buf.data(), fname_buf.size(),                 \
                           "OUTPUT_%d_%d.mp4", r.low, r.high);                 \
            macro_use_file(fname_buf.data());                                  \
            i = r.high + 1;                                                    \
        }                                                                      \
        while (i < (macro_nb_segments)) {                                      \
            (void)snprintf(fname_buf.data(), fname_buf.size(), "OUTPUT%d.mp4", \
                           i++);                                               \
            macro_use_file(fname_buf.data());                                  \
        }                                                                      \
    }

// TODO make SURE this code works man.
// F(uint32_t segment_index)
// G(ConcatRange segment_range)
template <typename F, typename G>
inline void iter_segs(F output_i, G output_range, uint32_t nb_segments,
                      std::span<ConcatRange> segs) {
    uint32_t i = 0;
    for (const auto& r : segs) {
        while (i < r.low) {
            output_i(i++);
        }
        output_range(r);
        i = r.high + 1;
    }
    while (i < nb_segments) {
        output_i(i++);
    }
}

inline std::vector<Segment> get_file_list(uint32_t nb_segments,
                                          std::span<ConcatRange> segs) {
    std::vector<Segment> segment_list{};
    // typical ratio of around 10% being segments
    segment_list.reserve(segs.size() * 10 + 8);
    iter_segs(
        [&](auto idx) { segment_list.emplace_back(idx, 0); },
        [&](auto range) { segment_list.emplace_back(range.low, range.high); },
        nb_segments, segs);
    return segment_list;
}

template <typename F>
inline void iter_segfiles(F use_file, uint32_t nb_segments,
                          std::span<ConcatRange> segs) {
    std::array<char, 64> fname_buf;
    iter_segs(
        [&](uint32_t i) {
            (void)snprintf(fname_buf.data(), fname_buf.size(), "OUTPUT%d.mp4",
                           i);
            use_file(fname_buf.data());
        },
        [&](ConcatRange r) {
            (void)snprintf(fname_buf.data(), fname_buf.size(),
                           "OUTPUT_%d_%d.mp4", r.low, r.high);
            use_file(fname_buf.data());
        },
        nb_segments, segs);
}

struct SegmentResult {
    std::vector<ConcatRange> concat_ranges{};
    std::vector<uint32_t> packet_offsets{};
};

// TODO clean up this API. It's a mess currently.
// segment video, including fixes to broken segments.
// also TODO error handling
// [[nodiscard]] inline std::vector<ConcatRange>
[[nodiscard]] inline SegmentResult
segment_video_fully(const char* url, unsigned int& nb_segments) {
    // TODO is there ANY way to optimize this allocation?
    // Do we really need FULLY RANDOM access?
    // Can we "reset" the buffer after a concatenation has been made or
    // something?
    SegmentResult res{};

    std::vector<Timestamp> timestamps{};
    timestamps.reserve(EST_NB_SEGMENTS * EST_PKTS_PER_SEG);
    // It would be nice to have both vectors somehow be a part of the same
    // larger allocation.
    DvAssert(segment_video(url, "OUTPUT%d.mp4", nb_segments, timestamps) == 0);

    printf("%zu - tss size (should be same as total packets)\n",
           timestamps.size());

    res.packet_offsets.reserve(EST_NB_SEGMENTS);

    // TODO: remove extra debugging checks for frames,
    // or make it optional or something (eventually).

    // TODO make sure no extra copies happen here
    res.concat_ranges =
        fix_broken_segments(nb_segments, res.packet_offsets, timestamps);

    return res;
}
