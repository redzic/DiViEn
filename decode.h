#pragma once

#include "resource.h"
#include "util.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <pthread.h>
#include <string_view>
#include <unistd.h>
#include <utility>
#include <variant>
#include <vector>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/log.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/rational.h>
}

// TODO possibly rename this for generalized libavcodec errors
struct DecoderCreationError {
    enum DCErrorType : uint8_t {
        AllocationFailure,
        NoVideoStream,
        NoDecoderAvailable,
        AVError,
    } type;
    int averror = 0;

    [[nodiscard]] constexpr std::array<char, AV_ERROR_MAX_STRING_SIZE>
    errmsg() const {
        static constexpr std::string_view errmsg_sv[] = {
            [AllocationFailure] = "Allocation Failure in decoder construction",
            [NoVideoStream] = "No video stream exists in input file",
            [NoDecoderAvailable] = "No decoder available for codec",
            // [AVError] = "",
        };

        // change this to the longest string in the array
        static_assert(errmsg_sv[AllocationFailure].size() <
                      AV_ERROR_MAX_STRING_SIZE);

        std::array<char, AV_ERROR_MAX_STRING_SIZE> errbuf{};

        if (averror) {
            DvAssert(averror != 0);
            av_make_error_string(errbuf.data(), errbuf.size(), averror);
        } else {
            DvAssert(averror == 0);
            auto msg = errmsg_sv[this->type];
            memcpy(errbuf.data(), msg.data(), msg.size());
            DvAssert(errbuf.size() > msg.size());
            // add null terminator
            errbuf[msg.size()] = '\0';
        }

        return errbuf;
    }
};

// copied from ffmpeg
inline void avframe_init(AVFrame* frame) {
    memset(frame, 0, sizeof(*frame));

    frame->pts = frame->pkt_dts = AV_NOPTS_VALUE;
    frame->best_effort_timestamp = AV_NOPTS_VALUE;
    frame->duration = 0;
    frame->time_base = (AVRational){0, 1};
    frame->sample_aspect_ratio = (AVRational){0, 1};
    frame->format = -1; /* unknown */
    frame->extended_data = frame->data;
    frame->color_primaries = AVCOL_PRI_UNSPECIFIED;
    frame->color_trc = AVCOL_TRC_UNSPECIFIED;
    frame->colorspace = AVCOL_SPC_UNSPECIFIED;
    frame->color_range = AVCOL_RANGE_UNSPECIFIED;
    frame->chroma_location = AVCHROMA_LOC_UNSPECIFIED;
    frame->flags = 0;
}

// Returns index of video stream, or -1 if it was not found.
// TODO replace this with av_find_best_stream() or whatever it's
// called.

// maybe add ctrl+C interrupt that just stops and flushes all packets so far?

using FrameBuf = std::vector<AVFrame>;

// yeah so even if you override the destructor, the other destructors
// still run afterward which is good.
struct DemuxerContext {
    DELETE_DEFAULT_CTORS(DemuxerContext)

    // TODO make sure this has no overhead
    std::unique_ptr<AVFormatContext, decltype([](AVFormatContext* ptr) {
                        avformat_close_input(&ptr);
                    })>
        demuxer;
    std::unique_ptr<AVPacket,
                    decltype([](AVPacket* ptr) { av_packet_free(&ptr); })>
        pkt;

    ~DemuxerContext() = default;
};

struct DecodeContext {
    // these fields can be null
    AVFormatContext* demuxer{nullptr};
    AVPacket* pkt{nullptr};

    AVCodecContext* decoder{nullptr};

    FrameBuf framebuf;

    // -1 if not initialized; else the index of the video stream
    int video_index = -1;

    // gets the index of the video stream.
    // Populates it if not available.
    // TODO error handling
    // If it returns -1, that means there was no video stream available.
    // TODO should figure out how to make wrapper class where
    // positive values indicates whatever, and negative values indicate
    // error. Ideally with zero overhead. Because variant will take up
    // many more bytes. Shouldn't be used for this kinda stuff.

    DELETE_DEFAULT_CTORS(DecodeContext)

    ~DecodeContext() {
        // Since we deleted all the copy/move constructors,
        // we can do this without handling a "moved from" case.

        for (auto& f : framebuf) {
            av_frame_unref(&f);
        }

        av_packet_free(&pkt);
        avformat_close_input(&demuxer);
        avcodec_free_context(&decoder);
    }

    // TODO maybe put another abstraction of just encapsulating
    // format context and wrap that one inside here
    // Takes ownership of the framebuf.
    // Be very careful when calling this constructor!
    explicit DecodeContext(AVFormatContext* demuxer_, AVCodecContext* decoder_,
                           AVPacket* pkt_, int vindex,
                           unsigned int framebuf_size)
        : demuxer(demuxer_), pkt(pkt_), decoder(decoder_), video_index(vindex) {
        // TODO: don't use vector for this.
        framebuf.reserve(framebuf_size);
        for (size_t i = 0; i < framebuf_size; i++) {
            framebuf.emplace_back();
        }
        for (auto& frame : framebuf)
            avframe_init(&frame);
    }

    // Open file and initialize video decoder.
    // TODO switch to std::expected
    [[nodiscard]] static std::variant<DecodeContext, DecoderCreationError>
    open(const char* url, unsigned int framebuf_size);
};

// how do you make a static allocation?

// Move cursor up and erase line

// Honestly this is kinda messed up.
// I think I should double check that all the ffmpeg errors have a negative
// value and stuff. Or just wrap all this in some kind of enum.

// framebuf_offset is the offset to start decoding into within its internal
// frame buffer.
// max_frames is the maximum number of frames that it should decode.
//
// Return value:
// < 0: an error occurred.
// > 0: Number of frames that were decoded.
//
// TODO: check if you have to copy the attributes to the function definition.
[[nodiscard]] int run_decoder(DecodeContext& dc, size_t framebuf_offset,
                              size_t max_frames);

struct DecodeContextResult {
    // This will just have nullptr fields
    DecodeContext dc;
    // if err.averror was non
    DecoderCreationError err;
};

// returns number of frames decoded
[[nodiscard]] inline int decode_loop(DecodeContext& dc) {

    int framecount = 0;
    while (true) {
        int ret = run_decoder(dc, 0, 1);
        if (ret <= 0) {
            break;
        }
        framecount += ret;
    }

    return framecount;
}

struct CountFramesResult {
    // TODO we need better signaling of this and whatnot
    // Probably make a struct where we have a separate
    // CountFramesResult and another thingy which is
    // the more detailed error information or whatever.

    // TODO check if order of this shit matters cuz I could kinda optimized
    // based on what we need to most probably access

    bool error_occurred : 1;
    unsigned int nb_discarded : 30;
    unsigned int frame_count : 31;
};

[[nodiscard]] CountFramesResult count_video_packets(DecodeContext& dc);

// returns 0 on success, or a negative number on failure.
int decode_next(DecodeContext& dc, AVFrame* frame);
