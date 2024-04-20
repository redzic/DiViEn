#pragma once

// include chunked encoding routines

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <span>

#include "decode.h"
#include "progress.h"

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
    std::atomic<uint32_t> nb_frames_skipped;

    chunk_hmap& resume_data;
    std::mutex resume_m;

    // ALL FIELDS BELOW ARE CONSTANT FIELDS
    unsigned int num_workers;
    unsigned int chunk_frame_size;
    unsigned int n_threads; // number of threads per worker

    const char* p_fname; // progress file path
    // must be initialized right after

    [[nodiscard]] AlwaysInline bool all_workers_finished() const noexcept {
        return this->nb_threads_done == this->num_workers;
    }

    DELETE_DEFAULT_CTORS(EncodeLoopState)

    EncodeLoopState(const char* filename, unsigned int num_workers_,
                    unsigned int chunk_frame_size_, unsigned int n_threads_,
                    chunk_hmap& resume_data_)
        : resume_data(resume_data_), num_workers(num_workers_),
          chunk_frame_size(chunk_frame_size_), n_threads(n_threads_),
          p_fname(filename) {}

    ~EncodeLoopState() = default;
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

// assume same naming convention
// this is direct concatenation, nothing extra done to files.
// hardcoded. TODO remove.
// perhaps we could special case this for 1 input file.
// TODO error handling
void raw_concat_files(std::string_view base_path, std::string_view prefix,
                      const char* out_filename, unsigned int num_files,
                      bool delete_after = false);

// TODO perhaps for tests we can try with lossless
// encoding and compare video results.
// perhaps the tests could use python scripts to call
// the binary or something.

// Maybe long term we could provide a C or C++ library.
// (probaby C).
// TODO move all the TODOs into a separate doc/file or something.

// decodes everything
// TODO need to make the output of this compatible with libav
// logging stuff. Maybe I can do that with a callback
// so that I can really handle it properly.
// This relies on global state. Do not call this function
// more than once.
// There's some kind of stalling going on here.
// TODO make option to test standalone encoder.
// There's only supposed to be 487 frames on 2_3.

// This function will create a base path.
// params is for encoder options

[[nodiscard]] int
chunked_encode_loop(EncoderOpts e_opts, const char* in_filename,
                    const char* out_filename, DecodeContext& d_ctx,
                    unsigned int num_workers, unsigned int chunk_frame_size,
                    unsigned int n_threads);