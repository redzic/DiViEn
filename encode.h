#pragma once

// include chunked encoding routines

#include <atomic>
#include <condition_variable>
#include <mutex>

#include "util.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/packet.h>
#include <libavfilter/avfilter.h>
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

AVPixelFormat av_pix_fmt_supported_version(AVPixelFormat pix_fmt);

AlwaysInline bool avframe_has_buffer(AVFrame* frame) {
    return frame->buf[0] != nullptr && frame->width > 0 && frame->height > 0;
}

struct EncodeLoopState {
    std::mutex global_decoder_mutex;
    unsigned int global_chunk_id = 0;

    // for printing progress
    std::condition_variable cv;
    // is mutex fast?
    // is memcpy optimized on windows? Memchr sure isn't.
    // is there a "runtime" that provides optimized implementations?
    // can I override the standard memcpy/memchr functions?
    std::mutex cv_m;

    std::atomic<uint32_t> nb_frames_done;
    std::atomic<uint32_t> nb_threads_done;

    // These 3 should never be modified after initialization.
    unsigned int num_workers;
    unsigned int chunk_frame_size;
    unsigned int n_threads; // number of threads per worker

    [[nodiscard]] AlwaysInline bool all_workers_finished() const noexcept {
        return this->nb_threads_done == this->num_workers;
    }

    DELETE_DEFAULT_CTORS(EncodeLoopState)
    ~EncodeLoopState() = default;

    explicit EncodeLoopState(unsigned int num_workers_,
                             unsigned int chunk_frame_size_,
                             unsigned int n_threads_)
        : num_workers(num_workers_), chunk_frame_size(chunk_frame_size_),
          n_threads(n_threads_) {}
};

// This is a REFERENCE to other existing data.
struct EncoderOpts {
    const char* encoder_name = nullptr;
    const char** params = nullptr;
    size_t n_param_pairs = 0;

    EncoderOpts() = delete;

    explicit constexpr EncoderOpts(const char* encoder_name_,
                                   const char** params_, size_t n_param_pairs_)
        : encoder_name(encoder_name_), params(params_),
          n_param_pairs(n_param_pairs_) {}
};

constexpr EncoderOpts DEFAULT_ENCODER = EncoderOpts("libx264", nullptr, 0);

struct EncoderContext {
    AVCodecContext* avcc{nullptr};
    AVPacket* pkt{nullptr};

    EncoderContext() = default;
    DELETE_COPYMOVE_CTORS(EncoderContext)

    // TODO proper error handling, return std::expected
    // caller needs to ensure they only call this once
    // The e_opts should start with a '-'.
    void initialize_codec(AVFrame* frame, unsigned int n_threads,
                          EncoderOpts e_opts);

    ~EncoderContext() {
        avcodec_free_context(&avcc);
        av_packet_free(&pkt);
    }
};

// If pkt is refcounted, we shouldn't have to copy any data.
// But the encoder may or may not create a reference.
// I think it probably does? Idk.
int encode_frame(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt,
                 FILE* ostream, std::atomic<uint32_t>& frame_count);

int encode_frames(const char* file_name, std::span<AVFrame*> framebuf,
                  EncodeLoopState& state, unsigned int n_threads,
                  EncoderOpts e_opts);

struct FrameAccurateWorkItem {
    // source chunk indexes (inclusive)
    uint32_t low_idx;
    uint32_t high_idx;
    // number of frames to skip at the beginning
    uint32_t nskip;
    // number of frames to decode after skipping
    uint32_t ndecode;

    template <size_t N> void fmt(std::array<char, N>& arr) {
        (void)snprintf(arr.data(), arr.size(), "[%d, %d], nskip %d, ndecode %d",
                       low_idx, high_idx, nskip, ndecode);
    }
};

void encode_frame_range(FrameAccurateWorkItem& data, const char* ofname);