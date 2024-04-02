// gonna have to obviously disable on some configurations and whatever
#define ASIO_HAS_IO_URING 1
#define ASIO_DISABLE_EPOLL 1

// pagecache + bypassing cache?

#include <asio.hpp>
#include <asio/buffer.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/read.hpp>
#include <asio/signal_set.hpp>
#include <asio/socket_base.hpp>
#include <asio/use_awaitable.hpp>
#include <asio/write.hpp>

// ------------------
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <mutex>
#include <pthread.h>
#include <span>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <variant>
#include <vector>

#include "decode.h"
#include "resource.h"
#include "segment.h"

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

// #define ASIO_ENABLE_HANDLER_TRACKING

#if defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__) &&       \
    defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)

/* little endian, supported */

#elif defined(__ORDER_LITTLE_ENDIAN__) && defined(__ORDER_BIG_ENDIAN__) &&     \
    defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_BIG_ENDIAN__)
#error                                                                         \
    "Big endian is not currently supported. Please file a bug on github for support."
// TODO when I get to this, test the code in qemu or something.
// But hopefully it shouldn't be too complicated.
// I'm pretty sure the byte order of the actual encoded data is always the same
// anyway.

#else

#error                                                                         \
    "Unsupported architecture or compiler. Please try using the latest version of clang. If that does not work, please file a bug on github."

#endif

namespace {

#define ERASE_LINE_ANSI "\x1B[1A\x1B[2K" // NOLINT
// #define ERASE_LINE_ANSI "" // NOLINT

#define AlwaysInline __attribute__((always_inline)) inline

AlwaysInline void w_err(std::string_view sv) {
    write(STDERR_FILENO, sv.data(), sv.size());
}

// so it seems like you have to call
// unref before you reuse AVPacket or AVFrame

void segvHandler(int /*unused*/) {
    w_err("Segmentation fault occurred. Please file a bug report on GitHub.\n");
    exit(EXIT_FAILURE); // NOLINT
}

// Idea:
// we could send the same chunk
// to different nodes if one is not doing it fast enough.
// To do this we could have some features in our vec of
//

// 128 chars is a good size I think
// Could even be reduced to 64
using fmt_buf = std::array<char, 128>;

template <class> inline constexpr bool always_false_v = false;

auto dist_ms(auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
        .count();
}

// TODO should we maybe pass by reference?
auto dist_us(auto start, auto end) {
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start)
        .count();
}

auto now() { return std::chrono::high_resolution_clock::now(); }

// for chunked encoding
struct EncodeLoopState {
    std::mutex global_decoder_mutex{};
    unsigned int global_chunk_id = 0;

    // for printing progress
    std::condition_variable cv{};
    // is mutex fast?
    // is memcpy optimized on windows? Memchr sure isn't.
    // is there a "runtime" that provides optimized implementations?
    // can I override the standard memcpy/memchr functions?
    std::mutex cv_m{};

    std::atomic<uint32_t> nb_frames_done{};
    std::atomic<uint32_t> nb_threads_done{};

    AlwaysInline bool all_workers_finished() {
        return this->nb_threads_done == NUM_WORKERS;
    }
};

// If pkt is refcounted, we shouldn't have to copy any data.
// But the encoder may or may not create a reference.
// I think it probably does? Idk.
int encode_frame(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt,
                 FILE* ostream) {
    // frame can be null, which is considered a flush frame
    DvAssert(enc_ctx != nullptr);
    int ret = avcodec_send_frame(enc_ctx, frame);
    if (ret < 0) {
        printf("error sending frame to encoder\n");
        return ret;
    }

    while (true) {
        int ret = avcodec_receive_packet(enc_ctx, pkt);
        // why check for eof though?
        // actually this doesn't really seem correct
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return 0;
        } else if (ret < 0) {
            printf("unspecified error during encoding\n");
            return ret;
        }

        // can write the compressed packet to the bitstream now

        // ostream.write(reinterpret_cast<char*>(pkt->data), pkt->size);
        (void)fwrite(pkt->data, pkt->size, 1, ostream);

        // just printing this takes 100ms...
        // printf("received packet from encoder of %d bytes\n", pkt->size);

        // WILL NEED THIS FUNCTION: av_frame_make_writable
        //
        // make_writable_frame actually COPIES the data over (potentially),
        // which is not ideal. if it's going to make a copy, the actual contents
        // don't need to be initialized. That's a super expensive thing to do
        // anyway.
        // I really don't get how that function works anyway.
        // It seems to actually delete the original anyway. So how does that
        // preserve references?
        // Unless there's a difference between av_frame_unref and
        // av_frame_free.

        av_packet_unref(pkt);
    }

    return 0;
}

AVPixelFormat av_pix_fmt_supported_version(AVPixelFormat pix_fmt) {
    switch (pix_fmt) {
    case AV_PIX_FMT_YUVJ420P:
        return AV_PIX_FMT_YUV420P;
    case AV_PIX_FMT_YUVJ422P:
        return AV_PIX_FMT_YUV422P;
    case AV_PIX_FMT_YUVJ444P:
        return AV_PIX_FMT_YUV444P;
    case AV_PIX_FMT_YUVJ440P:
        return AV_PIX_FMT_YUV440P;
    case AV_PIX_FMT_YUVJ411P:
        return AV_PIX_FMT_YUV411P;
    default:
        return pix_fmt;
    }
}

// TODO ensure that two clients don't run in the same directory.
// For now at least tell the user that they shouldn't do this.

// allocates entire buffer upfront
// single threaded version
// TODO maybe implement as callback instead

// TODO see if it's possible to reuse an encoder
// by calling avcodec_open2 again.

// we need to make a version of this that doesn't just encode everything and
// flush the output at once

struct EncoderContext {
    AVCodecContext* avcc{nullptr};
    AVPacket* pkt{nullptr};

    EncoderContext() = default;
    EncoderContext(EncoderContext&) = delete;
    EncoderContext(EncoderContext&&) = delete;
    EncoderContext& operator=(const EncoderContext&) = delete;
    EncoderContext& operator=(const EncoderContext&&) = delete;

    // TODO proper error handling, return std::expected
    // caller needs to ensure they only call this once
    void initialize_codec(AVFrame* frame) {
        // const auto* codec = avcodec_find_encoder_by_name("libaom-av1");
        const auto* codec = avcodec_find_encoder_by_name("libx264");
        avcc = avcodec_alloc_context3(codec);
        DvAssert(avcc);
        pkt = av_packet_alloc();
        DvAssert(pkt);
        avcc->thread_count = THREADS_PER_WORKER;
        // arbitrary values
        avcc->time_base = (AVRational){1, 25};
        avcc->framerate = (AVRational){25, 1};

        avcc->width = frame->width;
        avcc->height = frame->height;
        avcc->pix_fmt =
            av_pix_fmt_supported_version((AVPixelFormat)frame->format);

        // X264/5:
        av_opt_set(avcc->priv_data, "crf", "30", 0);
        av_opt_set(avcc->priv_data, "preset", "ultrafast", 0);
        // AOM:
        // av_opt_set(avcc->priv_data, "cpu-used", "6", 0);
        // av_opt_set(avcc->priv_data, "end-usage", "q", 0);
        // av_opt_set(avcc->priv_data, "cq-level", "30", 0);
        // av_opt_set(avcc->priv_data, "enable-qm", "1", 0);

        int ret = avcodec_open2(avcc, codec, nullptr);
        DvAssert(ret == 0 && "Failed to open encoder codec");
    }

    // TODO move encoding functions to this struct

    ~EncoderContext() {
        avcodec_free_context(&avcc);
        av_packet_free(&pkt);
    }
};

int encode_frames(const char* file_name, std::span<AVFrame*> frame_buffer,
                  EncodeLoopState& state) {
    DvAssert(!frame_buffer.empty());

    EncoderContext encoder;
    encoder.initialize_codec(frame_buffer[0]);

    // C-style IO is needed for binary size to not explode on Windows with
    // static linking

    // TODO use unique_ptr as wrapper resource manager
    make_file(file, file_name, "wb");

    for (auto* frame : frame_buffer) {
        // required
        frame->pict_type = AV_PICTURE_TYPE_NONE;
        encode_frame(encoder.avcc, frame, encoder.pkt, file.get());
        state.nb_frames_done++;
    }
    // need to send flush packet after we're done
    encode_frame(encoder.avcc, nullptr, encoder.pkt, file.get());

    return 0;
}

int encode_chunk(unsigned int chunk_idx, std::span<AVFrame*> framebuf,
                 EncodeLoopState& state) {
    std::array<char, 64> buf{};

    // this should write null terminator
    (void)snprintf(buf.data(), buf.size(), "file %d.mp4", chunk_idx);

    return encode_frames(buf.data(), framebuf, state);
}

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

// TODO: possible optimization idea.
// For segments that need very high nskip value,
// fall back to sequential model and just give.
// that entire chunk to one client.

// runs single threaded mode
void encode_frame_range(FrameAccurateWorkItem& data, const char* fname) {
    // TODO clean up string formatting, move it all to one centralized place
    // format is
    // "client_input_{idx}.mp4"

    // we only skip frames on the first chunk, otherwise it wouldn't
    // make any sense. All chunks have frames we need to decode.
    EncoderContext encoder;
    // now we have to encode exactly ndecode frames
    auto nleft = data.ndecode;

    make_file(efptr, fname, "wb");

    bool enc_was_init = false;
    uint32_t nframes_done = 0;

    printf("frame= 0\n");

    for (uint32_t idx = data.low_idx; idx <= data.high_idx; idx++) {
        fmt_buf input_fname;
        (void)snprintf(input_fname.data(), input_fname.size(),
                       "client_input_%d.mp4", idx);
        auto dres = DecodeContext::open(input_fname.data());

        // perhaps we could initialize all these decoders at the same time...
        // to save time.

        std::visit(
            [&, efptr = efptr.get()](auto&& dc) {
                using T = std::decay_t<decltype(dc)>;

                if constexpr (std::is_same_v<T, DecodeContext>) {
                    // main_encode_loop("output.mp4", arg);
                    int nb_decoded = 0;
                    // do we need to... uh...

                    for (auto* frame : dc.framebuf) {
                        av_frame_make_writable(frame);
                    }

                    // TODO split loop if possible
                    if (idx == data.low_idx) {
                        // means we first need to decode nskip frames
                        if (data.nskip > 0) {
                            DvAssert(run_decoder(dc, 0, 1) == 1);
                            encoder.initialize_codec(dc.framebuf[0]);
                            enc_was_init = true;
                            for (auto nf = data.nskip - 1; nf != 0; nf--) {
                                DvAssert(run_decoder(dc, 0, 1) == 1);
                            }
                        }
                    }

                    // TODO allow configurable frame size when constructing
                    // decoder, to avoid wasting memory
                    while (nleft > 0 &&
                           (nb_decoded = run_decoder(dc, 0, nleft)) > 0) {
                        if (!enc_was_init) [[unlikely]] {
                            encoder.initialize_codec(dc.framebuf[0]);
                            enc_was_init = true;
                        }

                        DvAssert(nb_decoded > 0);
                        nleft -= nb_decoded;

                        printf("Decoded %d frames\n", nb_decoded);

                        std::span<AVFrame*> frame_range(dc.framebuf.data(),
                                                        nb_decoded);

                        for (auto* frame : frame_range) {
                            // required
                            frame->pict_type = AV_PICTURE_TYPE_NONE;
                            encode_frame(encoder.avcc, frame, encoder.pkt,
                                         efptr);
                            printf(ERASE_LINE_ANSI "frame= %u\n",
                                   ++nframes_done);
                        }
                    }
                } else {
                    printf("Decoder failed to open for input file '%s'\n",
                           input_fname.data());
                }
            },
            dres);
    }

    DvAssert(efptr.get() != nullptr);
    encode_frame(encoder.avcc, nullptr, encoder.pkt, efptr.get());
}

// framebuf is start of frame buffer that worker can use
int worker_thread(unsigned int worker_id, DecodeContext& decoder,
                  EncodeLoopState& state) {
    while (true) {
        // should only access decoder once lock has been acquired
        // uh should we replace with like unique_lock or lock_guard
        // or something like that?
        // idk how save this is
        state.global_decoder_mutex.lock();

        // decode CHUNK_FRAME_SIZE frames into frame buffer
        int frames = run_decoder(decoder, worker_id * CHUNK_FRAME_SIZE,
                                 CHUNK_FRAME_SIZE);

        // error decoding
        if (frames <= 0) {
            state.nb_threads_done++;
            state.cv.notify_one();
            state.global_decoder_mutex.unlock();
            return frames;
        }

        // these accesses are behind mutex so we're all good
        auto chunk_idx = state.global_chunk_id++;
        // increment for next chunk

        // can assume frames are available, so unlock the mutex so
        // other threads can use the decoder
        state.global_decoder_mutex.unlock();

        // a little sketchy but in theory this should be fine
        // since framebuf is never modified
        int ret = encode_chunk(
            chunk_idx,
            {decoder.framebuf.data() + (worker_id * CHUNK_FRAME_SIZE),
             (size_t)frames},
            state);

        if (ret != 0) {
            // in theory... this shouldn't need to happen as this is an encoding
            // error
            // mutex was already unlocked so we don't unlock.

            state.nb_threads_done++;

            // in normal circumstances we return from infinite loop via decoding
            // error (which we expect to be EOF).
            state.cv.notify_one();

            return ret;
        }
    }
}

void avlog_do_nothing(void* /* unused */, int /* level */,
                      const char* /* szFmt */, va_list /* varg */) {}

// so I mean as hack we could do this dumb shit
// with checking the callback stuff...
// void av_log_TEST_CALLBACK(void* ptr, int level, const char* fmt, va_list vl)
// {
//     printf("%s\n", fmt);
// }

// assume same naming convention
// this is direct concatenation, nothing extra done to files.
// hardcoded. TODO remove.
// perhaps we could special case this for 1 input file.
void raw_concat_files(const char* out_filename, unsigned int num_files,
                      bool delete_after = false) {
    std::array<char, 64> buf{};

    std::ofstream dst(out_filename, std::ios::binary);

    for (unsigned int i = 0; i < num_files; i++) {
        (void)snprintf(buf.data(), buf.size(), "file %d.mp4", i);
        std::ifstream src(buf.data(), std::ios::binary);
        dst << src.rdbuf();
        src.close();
        // delete file after done
        if (delete_after) {
            DvAssert(std::remove(buf.data()) == 0);
        }
    }

    dst.close();
}

} // namespace

// clang AST transformations is something to consider
// for converting [] to .at()

// void encode_loop_nonchunked(DecodeContext& d_ctx) {}

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
void main_encode_loop(const char* out_filename, DecodeContext& d_ctx) {
    // uh... we really need a safe way to do this and stuff
    // this is really bad... just assuming enough frames are
    // available
    printf("Writing encoded output to '%s'\n", out_filename);

    EncodeLoopState state{};

    auto start = now();

    // TODO use stack array here
    // even if we have dynamic workers, I believe we
    // can use alloca. Not really totally ideal but,
    // whatever. Or just array with max capacity.
    // std::vector<std::thread> thread_vector{};
    // thread_vector.reserve(NUM_WORKERS);
    std::array<std::thread, NUM_WORKERS> thread_vector;

    // spawn worker threads
    for (unsigned int i = 0; i < NUM_WORKERS; i++) {
        // TODO I think it's theoretically possible
        // that constructing the threads fail, in which case
        // the memory needs to be cleaned up differently (I think).
        // Since otherwise we'd be destructing unallocated objects.
        new (&thread_vector[i])
            std::thread(&worker_thread, i, std::ref(d_ctx), std::ref(state));
    }

    printf("frame= 0  (0 fps)\n");
    uint32_t last_frames = 0;

    auto compute_fps = [](uint32_t n_frames, int64_t time_ms) -> double {
        if (time_ms <= 0) [[unlikely]] {
            return INFINITY;
        } else [[likely]] {
            return static_cast<double>(1000 * n_frames) /
                   static_cast<double>(time_ms);
        }
    };

    // TODO minimize size of these buffers
    // TODO I wonder if it's more efficient to join these buffers
    // into one. And use each half.
    std::array<char, 32> local_fps_fmt;
    std::array<char, 32> avg_fps_fmt;

    while (true) {
        // acquire lock on mutex I guess?
        // TODO: see if we can release this lock earlier.
        std::unique_lock<std::mutex> lk(state.cv_m);

        // so for some reason this notify system isn't good enough
        // for checking if all threads have completed.
        auto local_start = now();
        // TODO Is this guaranteed to deal with "spurious wakes"?
        // The documentation does specfically say it can be woken spuriously.
        // but that was for wait not wait_for.
        auto status = state.cv.wait_for(lk, std::chrono::seconds(1));
        // bruh where should we unlock this
        // lk.unlock();

        auto n_frames = state.nb_frames_done.load();
        auto frame_diff = n_frames - last_frames;
        last_frames = n_frames;
        // since we waited exactly 1 second, frame_diff is the fps,
        // unless the status is no_timeout.

        auto local_now = now();
        auto total_elapsed_ms = dist_ms(start, local_now);

        // So this part of the code can actually run multiple times.
        // For each thread that signals completion.
        // Well this does work for avoiding extra waiting unnecessarily.
        // TODO simplify/optimize this code if possible

        if (status == std::cv_status::no_timeout) [[unlikely]] {
            // this means we didn't wait for the full time
            auto elapsed_local_ms = dist_ms(local_start, local_now);
            if (elapsed_local_ms > 0) [[likely]] {
                auto local_fps = static_cast<int32_t>(
                    compute_fps(frame_diff, elapsed_local_ms));
                (void)snprintf(local_fps_fmt.data(), local_fps_fmt.size(), "%d",
                               local_fps);
            } else {
                // write ? to buffer
                // TODO convert everything to a better mechanism not involving
                // null terminators and printf
                // 2 bytes including null terminator
                memcpy(local_fps_fmt.data(), "?", 2);
            }
        } else {
            // we waited the full 1s
            // TODO ensure it's actually 1s in case of spurious wakes
            // auto local_fps = frame_diff;
            (void)snprintf(local_fps_fmt.data(), local_fps_fmt.size(), "%d",
                           frame_diff);
        }
        // average fps from start of encoding process
        // TODO can we convert to faster loop with like boolean flag + function
        // pointer or something? It probably won't actually end up being faster
        // due to overhead tho.
        if (total_elapsed_ms == 0) [[unlikely]] {
            memcpy(avg_fps_fmt.data(), "?", 2);
        } else [[likely]] {
            auto avg_fps = compute_fps(n_frames, total_elapsed_ms);
            snprintf(avg_fps_fmt.data(), avg_fps_fmt.size(), "%.1f", avg_fps);
        }

        // print progress
        // TODO I guess this should detect if we are outputting to a
        // terminal/pipe and don't print ERASE_LINE_ASCII if not a tty.
        printf(ERASE_LINE_ANSI "frame= %d  (%s fps avg, %s fps curr)\n",
               n_frames, avg_fps_fmt.data(), local_fps_fmt.data());

        static_assert(NUM_WORKERS >= 1);

        if (state.all_workers_finished()) {
            break;
        }
    }

    // In theory all the threads have already exited here
    // but we need to call .join() anyways.
    for (auto& t : thread_vector) {
        t.join();
        // should we call this? I'm not sure.
        // asan/ubsan doesn't complain but at the same time
        // aren't the destructors already called?
        // and from some tests it does seem like
        // you can easily do a double free if you want to...
        // t.~thread();
    }

    // there is no active lock on the mutex since all threads
    // terminated, so global_chunk_id can be safely accessed.
    // raw_concat_files(out_filename, global_chunk_id, true);
    // TODO why does deleting files fail sometimes?
    raw_concat_files(out_filename, state.global_chunk_id, false);
}

// I guess the next step is to send each chunk in a loop.

// each index tells you the packet offset for that segment
// for both dts and pts, of course.

// this code is so incredibly messy bro

// I guess maybe this should iterate backwards
// from max index to 0, that way when a thing happens.
// we can avoid double counting the segments. oR wait...
// maybe not.

// basically what we need need to do
// is build these vectors
// TODO move this into segment.h

// I think we just need to configure the AVOutputFormat propertly.
// It seems like if this is null, the output format is guessed
// based on the file extension.

// TODO: replace manual loop over streams with av_find_best_stream or whatever

// For concat I really don't know if I should use the concat
// filter or just directly put the packets next to each other.
// Probably better to use concat filter I guess.

// BE CAREFUL NOT TO WRITE ASSERTS
// WITH SIDE EFFECTS.

// so now that we got the segmenting code working I guess it's time to
// write up the TCP server and client.

// TODO We should add a test for this, and measure "decoder stalling".
// There's also a problem of what if we need really long segments plus short
// segments. Then our current model of using one global decoder isn't so good,
// also because we would have to allocate the entire chunk ahead of time.
// which would be a huge memory problem obviously. It works ok with fixed sized
// chunks though.

// I think there's a solution to fix the memory hogging problem,
// but it would still require having the decoder available.
//      (the fix is to decode n frames at a time and cycling
//      between encoder and decoder instead of full decode + Encode step.)

// so unique_ptr DOES seem to be "zero cost" if you use make_unique.
// in certain cases at least. and noexcept.

// TODO: if I REALLY wanna make this code complex,
// I could implement some kind of way to decode packets
// as they are coming in instead of waiting for the entire chunk to
// be received first.

// also I should check for if

// https://github.com/facebook/wdt
// perhaps we should consider this.

// maybe we can statically link agner fog
// libraries too.

// Perhaps we should just use one global (netowkr) reader, to minimize latency.
// But hmm, perhaps that's not the best approach either, because one client
// may not have the best data connection speed.
// The best approach would be rather complex.
// We should at least be maxing out our bandwidth.

// we could also possibly look into using bittorrent.

using asio::awaitable;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
using asio::use_future;
using asio::ip::tcp;

using tcp_acceptor =
    asio::basic_socket_acceptor<asio::ip::tcp, asio::io_context::executor_type>;

using tcp_socket =
    asio::basic_stream_socket<asio::ip::tcp, asio::io_context::executor_type>;

// TODO check if sanitizers can catch these bugs

// in theory passing a reference here should be safe because
// it always comes from `echo`, which is owned
// awaitable<void> echo_once(tcp_socket& socket, int i) {
//     char data[128];
//     std::size_t n =
//         co_await socket.async_read_some(asio::buffer(data),
//         asio::redirect_error(use_awaitable, error));
//     printf("[i=%d] Read %d bytes\n", i, (int)n);
//     co_await async_write(socket, asio::buffer(data, n),
//     asio::redirect_error(use_awaitable, error));
// }

// echo LOOP

// alright so THANKFULLY this seems to automatically work with
// multiple connections, it's not blocking on only one thing.

// TODO.
// perhaps we could avoid the overhead of redundantly reading from files.
// By just storing the compressed data in memory. (AVIOContext)

// I guess now the step is to write the loop so that we interleave decoding
// and encoding, so that we decouple the framebuf size and all that.
// But we should double check that the encoder doesn't rely on original
// decoded frames existing. Or that refcounting handles them properly or
// whatever.

// TODO make this use the parsing mechanism with headers + EOF indication
// so that we can send multiple files through this.
// Also make a version of this that writes the file instead of reading.

// ========================
//
// Here's the protoocol.
// 4 byte header. (int). unsigned int,
//      - LITTLE ENDIAN.
//        Should make this a function.
// either positive value.
//  or 0, which means that's the end of the stream.
//
// ========================

// yeah so we should DEFINITELY use a bigger buffer size than 2048 bytes lol.

// TODO does all the other code work if we receive a size
// bigger than what the buffer can handle? Don't we need some
// loops in that case
constexpr size_t TCP_BUFFER_SIZE = 2048z * 32 * 4; // 256 Kb buffer

// TODO move networking code to another flie.

// Returns number of bytes read.
// TODO make async.
[[nodiscard]] awaitable<size_t>
socket_send_file(tcp_socket& socket, const char* filename,
                 asio::error_code& error) { // Open a file in read mode

    printf("Called socket_send_file\n");

    make_file(fptr, filename, "rb");
    if (!filename) {
        printf("Opening file %s failed\n", filename);
    }
    DvAssert(fptr.get());

    std::array<uint8_t, TCP_BUFFER_SIZE> read_buf;
    uint32_t header = 0;
    // TODO stop ignoring errors

    // if return value
    // TODO error handling
    size_t n_read = 0;
    size_t total_read = 0;
    // TODO perhaps just reuse n_read variable in case the compiler doesn't
    // realize
    while ((n_read = fread(read_buf.data(), 1, read_buf.size(), fptr.get())) >
           0) {
        // this probably doesn't even work man.
        // I believe this should work.
        // TODO FIX ENDIANNESS!!!!
        // TODO error handling
        header = n_read;
        DvAssert(co_await asio::async_write(
                     socket, asio::buffer(&header, sizeof(header)),
                     asio::redirect_error(use_awaitable, error)) == 4);
        // TODO: check if destructors properly run with co_return.
        // They probably do but just making sure.
        DvAssert(co_await asio::async_write(
                     socket, asio::buffer(read_buf, n_read),
                     asio::redirect_error(use_awaitable, error)) == n_read);
        // printf("Wrote %zu bytes\n", n_read);

        total_read += n_read;
    }
    header = 0;
    DvAssert(co_await asio::async_write(
                 socket, asio::buffer(&header, sizeof(header)),
                 asio::redirect_error(
                     asio::redirect_error(use_awaitable, error), error)) == 4);

    co_return total_read;
}

// TODO check on godbolt if the compiler auto dedups
// these calls, or if it inlines them or what.
// I imagine it prob doesn't. But who knows. I mean .
// Yeah probably not. But we should check
// how to make it dedup it.
template <typename Functor> awaitable<size_t> print_transfer_speed(Functor f) {
    auto start = now();
    size_t bytes = co_await f();

    if (bytes != 0) {
        auto elapsed_us = dist_us(start, now());
        // megabytes per second
        auto mb_s =
            static_cast<double>(bytes) / static_cast<double>(elapsed_us);

        // TODO : to make this more accurate, only count
        // the times we are actually waiting on the file (not disk write
        // times)
        printf(" [%.3f ms] %.1f MB/s throughput (%.0f Mbps)\n",
               static_cast<double>(elapsed_us) * 0.001, mb_s, 8.0 * mb_s);
    }
    co_return bytes;
}

// does directly returning the awaitable from here also work?
// I mean, it seemed to.
// Unfortunately, this causes overhead because we're passing
// a coroutine lambda. I guess it should be negligible mostly but still.
#define DISPLAY_SPEED(arguments)                                               \
    print_transfer_speed(                                                      \
        [&]() -> awaitable<size_t> { co_return co_await (arguments); })

// TODO come up with a better name for this function and the write version
// This is just an ugly name man.
// Returns number of bytes received
[[nodiscard]] awaitable<size_t> socket_recv_file(tcp_socket& socket,
                                                 const char* dumpfile,
                                                 asio::error_code& error) {
    printf("Called socket_recv_file\n");

    // TODO ideally we don't create any file unless we at least read the header
    // or something. I think that wouldn't even be hard to do actually.
    make_file(fptr, dumpfile, "wb");
    if (fptr == nullptr) {
        // technically this isn't thread safe
        // and I mean it is actually possible to access this concurrently
        // in an unsafe way... but whatever
        printf("fopen() error: %s\n", strerror(errno));
        co_return 0;
    }

    DvAssert(fptr.get() != nullptr);
    size_t written = 0;
    uint32_t header = 0;
    while (true) {
        // TODO figure out optimal buffer size
        std::array<uint8_t, TCP_BUFFER_SIZE> buf;

        // it seems like the issue is that once the file has been received,
        // this doesn't sufficiently block for the data to be receieved from the
        // client
        // we should definitely like, handle connection closes properly.
        size_t nread = co_await asio::async_read(
            socket, asio::buffer(&header, sizeof(header)),
            asio::redirect_error(use_awaitable, error));

        if (error == asio::error::eof) {
            // break;
            printf("received eof\n");
            co_return 0;
        }

        if (header == 0) {
            // printf("header was 0 (means end of data stream according to "
            //        "protocol).\n");
            break;
        }
        DvAssert(nread == 4);

        size_t len = 0;
        // TODO make a wrapper for this man.
        // TODO yeah pretty sure if header > buf.size(),
        // this doesn't read all the bytes.
        DvAssert(header <= buf.size());
        DvAssert((len = co_await asio::async_read(
                      socket, asio::buffer(buf, header),
                      asio::redirect_error(use_awaitable, error))) == header);

        // now we are no longer relying on the connection closing via eof
        if (error == asio::error::eof) {
            printf("UNEXPECTED EOF\n");
            co_return 0;
        } else if (error) {
            throw asio::system_error(error); // Some other error.}
        }

        // printf("Read %zu bytes\n", len);

        // assume successful call (in release mode)
        DvAssert(fwrite(buf.data(), 1, len, fptr.get()) == len);
        written += len;
    }

    co_return written;
}

// new protocol, receive and send N chunks
// When server closes connection, client closes also.
// I Guess we don't have to actually change anything.

// TODO (long term) Honestly using bittorrent to distribute the chunks
// among other stuff is probably the way to go.
// TODO is it possible to move ownership and return it back? Without a ton of
// overhead?
// TODO remove context parameter

// I think we also need to pass a condition variable here.
// perhaps we should just put all of this in one struct.
// This is getting really complicated man.

// handle_conn will signify every time a chunk is done, I guess.
// Or perhaps we can only signal if the total chunks done equals the

// owned data

// TODO
// if we allow client sending error messages back,
// limit the maximum size we receive.

// TODO: when decoding skipped frames,
// put them all in the same avframe.

// for distributed encoding, including fixed segments,
// arbitrary file access
// TODO need to deduplicate this code and instead of using
// 0 as special value for high_idx, just store the same
// index as low_idx. It would simplify code a good amount.

struct FinalWorklist {
    // This is supposed to be a list of actual original segments
    // Which could either be an individual segment or a concatenated one.
    std::vector<FixedSegment> source_chunks;
    // selects low and high indexes of source_chunks
    std::vector<FrameAccurateWorkItem> client_chunks;
};

struct ServerData {
    uint32_t orig_work_size;
    std::atomic<uint32_t> chunks_done;
    FinalWorklist work;
    std::mutex work_list_mutex;

    // for thread killing
    std::condition_variable tk_cv;
    std::mutex tk_cv_m;
};

// like number of chunks that there are.

[[nodiscard]] awaitable<void> handle_conn(asio::io_context& context,
                                          tcp_socket socket,
                                          unsigned int conn_num,
                                          ServerData& state) {
    try {

        printf("Entered handle_conn\n");
        // so we need to signal to the main thread that we are uh
        // waiting and shit.

        // So it's actually quite simple.
        // Instead of sending all chunks, we send chunks from a queue.
        // That's basically it.
        // There's also a possibility that we add stuff back to the queue.
        // I mean tbf we can do that with mpsc as well.
        // TODO Mutex<Vector> is probably not the most efficient approach.
        // Maybe we should actually use a linked list or something.
        // Find something better if possible.

        asio::error_code error;

        // bro so much stuff needs to be refactored for this
        // the output names are all mumbo jumboed

        for (;;) {
            FrameAccurateWorkItem work{};
            {
                std::lock_guard<std::mutex> guard(state.work_list_mutex);
                // we can now safely access the work list

                // no more work left
                if (state.work.client_chunks.empty()) {

                    printf("No more work left\n");
                    // Just because the work list is empty doesn't mean
                    // the work has been completed. It just means
                    // it's been ALLOCATED (distributed).
                    // And we really shouldn't use context.stop() either...
                    // so I think socket only shuts down the current one
                    // socket.shutdown(asio::socket_base::shutdown_receive);
                    // yeah we can't just do this..
                    // we would have to make sure ALL OTHER
                    // clients are stopped too...
                    // context.stop();
                    co_return;
                }

                // get some work to be done
                work = state.work.client_chunks.back();

                state.work.client_chunks.pop_back();
            }

            fmt_buf tmp;
            work.fmt(tmp);

            // we are just going to send each thing in order
            // low, high, nskip, ndecode
            co_await asio::async_write(socket, asio::buffer(&work.low_idx, 4),
                                       use_awaitable);
            co_await asio::async_write(socket, asio::buffer(&work.high_idx, 4),
                                       use_awaitable);
            co_await asio::async_write(socket, asio::buffer(&work.nskip, 4),
                                       use_awaitable);
            co_await asio::async_write(socket, asio::buffer(&work.ndecode, 4),
                                       use_awaitable);

            // now we wait for the client to tell us which chunks out of those
            // it actually needs the format for that is 4 byte length, followed
            // by vec of 4 byte segments. Each 4 bytes tells which index it
            // needs.
            // TODO for validate_worker or whatever, make sure client can only
            // request chunks that we specified

            DvAssert(work.low_idx <= work.high_idx);

            fmt_buf fname;
            for (uint32_t i = work.low_idx; i <= work.high_idx; i++) {
                // client is supposed to tell us if it already has this chunk or
                // not.

                // just a yes or no
                uint8_t client_already_has_chunk = 0;
                co_await asio::async_read(
                    socket, asio::buffer(&client_already_has_chunk, 1),
                    use_awaitable);

                if (client_already_has_chunk) {
                    continue;
                }

                // this should never be modified so it should be ok to just
                // access this without a mutex oh ok so here the problem is
                // we are using data that is not meant to be used the way we
                // are using it... I think the indexes are mismatched. But
                // we do need to fix this.
                printf(" --- INDEX %d\n", i);
                // state.work.source_chunks.at(i).fmt(fname);
                state.work.source_chunks.at(i).fmt(fname);

                // we are uploading this
                printf("Sending '%s' to client #%d\n", fname.data(), conn_num);
                auto bytes1 =
                    co_await socket_send_file(socket, fname.data(), error);
                printf("Sent %zu bytes to client\n", bytes1);
            }

            // send multiple files, receive one encoded file

            // TODO. It would be faster to transfer the encoded packets
            // as they are complete, so we do stuff in parallel.
            // Instead of waiting around for chunks to be entirely finished.
            // I think we can even do this without changing the protocol.

            // Receive back encoded data
            // TODO the display on this is totally misleading
            // because it takes into account the encoding time as well.
            // Fixing this would require a redesign to the protocol I guess.

            // TODO add verify work option or something, ensures packet count is
            // what was expected. Or you could call it trust_workers or
            // something. which is set to false by default. Or whatever.
            fmt_buf recv_buf;
            (void)snprintf(recv_buf.data(), recv_buf.size(),
                           "recv_client_%u%u%u%u.mp4", work.low_idx,
                           work.high_idx, work.ndecode, work.nskip);
            auto bytes =
                co_await socket_recv_file(socket, recv_buf.data(), error);
            printf("Read back %zu bytes [from client #%d]\n", bytes, conn_num);

            // here we receive encoded data
            std::lock_guard<std::mutex> lk(state.tk_cv_m);
            state.chunks_done++;
            // not entirely sure if this lock is really necessary
            state.tk_cv.notify_one();
        }

        // unfortunately this is the only real solution to this problem
        // ITER_SEGFILES(co_await use_file, nb_segments, seg_result);

        // should never be reached.
        co_return;
    } catch (std::exception& e) {
        // e.what()
        printf("exception occurred in handle_conn(): %s\n", e.what());
    }
}

// TODO do not use any_io_executor as it's the defualt polymorphic
// type
// try to use io_context::executor_type instead.

// memory leaks happen when I do ctrl+C...
// Is that supposed to happen?

// also TODO need to add mechanism to shutdown server

// there's an issue here man.
awaitable<void> kill_context(asio::io_context& context) {
    context.stop();

    co_return;
}

// alright well this does seem to work
// and the io_context
// is there a better approach though?
// kills the io_context when all work is finished.
// https://en.cppreference.com/w/cpp/thread/condition_variable/wait
void server_stopper_thread(asio::io_context& context, ServerData& data) {
    std::unique_lock<std::mutex> lk(data.tk_cv_m);
    data.tk_cv.wait(lk,
                    [&]() { return data.orig_work_size == data.chunks_done; });

    co_spawn(context, kill_context(context), detached);
}

// god this code is messy
// TODO return nb_segments as value (and all similar functions in code)
// sadly there's some kind of bug with segmenting that huge video.
// TODO it might be because we're not using av_make_frame_writable.
// Check if decoder is failing or just encoder.
FinalWorklist server_prepare_work(const char* source_file,
                                  unsigned int& nb_segments) {
    unsigned int n_segments = 0;
    auto sg = segment_video_fully(source_file, n_segments);
    nb_segments = n_segments;

    DvAssert(!sg.packet_offsets.empty());

    auto fixed_chunks = get_file_list(nb_segments, sg.concat_ranges);
    // TODO optimize this. We don't have to recompute the packet amounts
    // (assuming the segmenting worked as expected)

    // create chunk list
    // bruh this is so damn messy
    // ideally we should use another type for this because these are actually
    // frame indexes not something else.
    std::vector<FixedSegment> scene_splits{};
    constexpr uint32_t SPLIT_SIZE = 250;
    for (uint32_t i = 0; i < sg.packet_offsets.back(); i += SPLIT_SIZE) {
        scene_splits.emplace_back(
            i, std::min(i + SPLIT_SIZE - 1, sg.packet_offsets.back() - 1));
    }

    printf("SCENE SPLITS\n");
    for (auto scene : scene_splits) {
        printf("[%d, %d], ", scene.low, scene.high);
    }
    printf("\n");

    // bruh now we need to do the O(n) version of the algorithm...
    // basically algorithm is just, if
    // but the thing is...
    // for one scene we might have like multiple splits or whatever
    // uhhh...
    // But it will NEVER be possible for the next split to go BACK
    // a segment, I believe. It might use the same but will NEVER
    // go back.
    // for now we are operating over original unconcatenated chunks.
    // But whatever...

    // work list based on scene segments

    std::vector<FrameAccurateWorkItem> work_items{};
    work_items.reserve(scene_splits.size() + 32);

    // TODO do this without copying the data over (in place)
    // would iterating in reverse help with that?
    std::vector<uint32_t> fixed_packet_offs{};
    fixed_packet_offs.reserve(sg.packet_offsets.size());
    iter_segs(
        [&](uint32_t i) { fixed_packet_offs.push_back(sg.packet_offsets[i]); },
        [&](ConcatRange r) {
            DvAssert(r.high > r.low);
            fixed_packet_offs.push_back(sg.packet_offsets[r.low]);
        },
        nb_segments, sg.concat_ranges);
    fixed_packet_offs.push_back(sg.packet_offsets.back());

    auto print_chunk_i = [&](auto chunk_idx) {
        printf("  Chunk i=%zu [%d, %d]\n", chunk_idx,
               fixed_packet_offs[chunk_idx],
               fixed_packet_offs[chunk_idx + 1] - 1);
    };
    auto chunki_maxidx = [&](auto chunk_idx) {
        return fixed_packet_offs[chunk_idx + 1] - 1;
    };

    auto overlap_exists = [&](auto chunk_idx, FixedSegment scene) {
        auto is_overlapping = [](auto x1, auto x2, auto y1, auto y2) {
            auto res = std::max(x1, y1) <= std::min(x2, y2);
            // printf("OVERLAP ? %d [%d, %d], [%d, %d]\n", (int)res, x1, x2, y1,
            //        y2);
            return res;
        };
        auto c_low_idx = fixed_packet_offs[chunk_idx];
        auto c_high_idx = chunki_maxidx(chunk_idx);
        // printf("    [ci %zu] ", chunk_idx);
        return is_overlapping(c_low_idx, c_high_idx, scene.low, scene.high);
    };

    // chunk idx may or may not increment.
    // size_t chunk_idx = 0;
    // yeah so the bug is here...
    // we need to fix it
    // and make a new packet offset
    for (auto scene : scene_splits) {
        // printf("[%d, %d] (ci %zu)\n", scene.low, scene.high, chunk_idx);
        printf("[%d, %d]\n", scene.low, scene.high);

        // Optimized version of code:
        // (Not entirely sure if it works in all cases but it seems to so far.)

        // loop_begin:
        //     if (chunk_idx + 1 >= sg.packet_offsets.size()) {
        //         break;
        //     }

        //     if (chunk_idx + 2 >= sg.packet_offsets.size()) {
        //         print_chunk_i(chunk_idx);
        //         break;
        //     }

        //     bool curr = overlap_exists(chunk_idx, scene);
        //     bool next = overlap_exists(chunk_idx + 1, scene);

        //     if (curr && next) {
        //         print_chunk_i(chunk_idx);
        //         // print_chunk_i(chunk_idx + 1);
        //         chunk_idx += 1;
        //         goto loop_begin;
        //     } else if (!curr && next) {
        //         print_chunk_i(chunk_idx + 1);
        //         chunk_idx += 2;
        //         goto loop_begin;
        //     } else if (curr && !next) {
        //         print_chunk_i(chunk_idx);
        //     } else {
        //         // !curr && !next
        //         // DvAssert(false);
        //         printf("[ci = %zu] Should not happen\n", chunk_idx);
        //     }

        // this is the "true" work list or whatever

        uint32_t low_idx = 0;
        uint32_t high_idx = 0;
        bool found_yet = false;
        // uint32_t
        size_t decode_nskip = 0;
        for (size_t i = 0; i < fixed_packet_offs.size() - 1; i++) {
            if (overlap_exists(i, scene)) {
                print_chunk_i(i);
                if (!found_yet) {
                    // means this is the first chunk
                    low_idx = i;
                    decode_nskip = scene.low - fixed_packet_offs[i];
                }
                high_idx = i;
                found_yet = true;
            } else if (found_yet) {
                break;
            }
        }
        work_items.push_back(FrameAccurateWorkItem{
            .low_idx = low_idx,
            .high_idx = high_idx,
            .nskip = (uint32_t)decode_nskip,
            .ndecode = scene.high - scene.low + 1,
        });
        printf("   nskip = %zu, ndecode = %u - range [%d, %d]\n", decode_nskip,
               scene.high - scene.low + 1, low_idx, high_idx);
    }

    // TODO optimize moving and initialiation of vectors if possible
    auto res = FinalWorklist{.source_chunks = std::move(fixed_chunks),
                             .client_chunks = std::move(work_items)};

    fmt_buf buf;
    // for (auto x : res.source_chunks) {
    //     x.fmt(buf);
    //     printf("%s\n", buf.data());
    // }
    // yeah so client_chunks should be based on new data bruh.
    // Not old one.
    for (auto x : res.client_chunks) {
        x.fmt(buf);
        printf("%s\n", buf.data());
    }

    return res;
}

// TODO remove unused arguments
awaitable<void> run_server(asio::io_context& context, const char* source_file,
                           ServerData& state) {

    asio::error_code error;

    tcp_acceptor acceptor(context, {tcp::v4(), 7878});

    printf("[Async] Listening for connections...\n");
    // for (unsigned int conn_num = 1; conn_num <= 3; conn_num++) {
    for (unsigned int conn_num = 1;; conn_num++) {
        // OH MY GOD. This counts as work for io_context!
        // So in theory we can remove the .stop() ont he context.

        // how can we handle either waiting for the socket, or
        // like waiting for the tasks to be finished

        // the core issue is that we need to be able to cancel the async_accept
        // which we will do by doing the io_context...
        // oh ok I remember by original idea.
        // Spawn another coroutine
        // We don't need to change anything here.

        // is it possible to do this without waiting though?
        tcp_socket socket = co_await acceptor.async_accept(
            asio::redirect_error(use_awaitable, error));

        if (error) {
            printf("Error connecting to client #%d: %s\n", conn_num,
                   error.message().c_str());
            continue;
        }

        printf("[TCP] Connection %d accepted\n", conn_num);
        // next step: detect when all work has been done somehow

        // man this is still gonna be a lot of work left...

        // uhh...
        // this could actually be REALLY bad.
        // since we just co_return.
        // or wait...
        // I think this works because we just infinitely wait for connections
        // so the other data doesn't go out of scope or anything.
        // Ideally we should add some kind of mechanism to track when stuff
        // is
        // finished.
        co_spawn(context,
                 handle_conn(context, std::move(socket), conn_num, state),
                 detached);

        printf("[TCP] Connection %d closed\n", conn_num);
    }

    printf("Returning from run_server()\n");
    co_return;
}

// So the client doesn't even have to be async.
// that can just be regular old sync.

template <typename F> struct OnReturn {
    F f;

    explicit OnReturn(F f_) : f(f_) {}
    OnReturn(const OnReturn&) = delete;
    OnReturn(OnReturn&&) = delete;
    ~OnReturn() { f(); }
};

// TODO. Perhaps the client should run, because it might lower throughput.
awaitable<void> run_client(asio::io_context& io_context, tcp_socket socket,
                           asio::error_code& error) {
    OnReturn io_context_stopper([&]() { io_context.stop(); });

    // TODO perhaps use flat array, should be faster since wouldn't need any
    // hashing.
    std::unordered_set<uint32_t> stored_chunks{};

    printf("Connected to server\n");

    // is there a way to design the protocol without having to know the
    // exact file sizes in bytes ahead of time?
    // because that would kinda just add totally unnecessary overhead
    // for nothing.

    // actually yes I think we can do that with a "state machine".
    // Once we get a message to start receiving file,
    // then that's when we transition to "receiving file" state.
    // each message is prefixed with a length then and then
    // we receive a final message that says that waas the last chunk.

    // and then the client sends back a response to each request,
    // saying "ok". If we don't receive the message, then
    // we report that I guess.

    // TODO I'm gonna need some kind of system to ensure server/client
    // code is always in sync. Probably just tests alone will get us
    // most of the way there.

    // we are receiving file from server here

    // TODO I need to detect when the client sends some nonsense.
    // Rn there are no checks.

    // TODO fix gcc warnings.

    // where to dump the input file from the server
    fmt_buf input_buf;
    for (;;) { //    change of protocol.
        // All subsequent headers are for the actual file data.

        // read multiple payloads from server
        FrameAccurateWorkItem work{};

        co_await asio::async_read(socket, asio::buffer(&work.low_idx, 4),
                                  use_awaitable);
        co_await asio::async_read(socket, asio::buffer(&work.high_idx, 4),
                                  use_awaitable);
        co_await asio::async_read(socket, asio::buffer(&work.nskip, 4),
                                  use_awaitable);
        co_await asio::async_read(socket, asio::buffer(&work.ndecode, 4),
                                  use_awaitable);

        DvAssert(work.low_idx <= work.high_idx);

        // why does it only send one chunk at a time though?
        // TODO add that mechanism of back and forth "do you already have this
        // file" and then only send necessary chunks

        fmt_buf tmp;
        work.fmt(tmp);
        printf(" header data: %s\n", tmp.data());

        // change of plans
        // server will ask for each chunk if it already has it
        for (uint32_t chunk_idx = work.low_idx; chunk_idx <= work.high_idx;
             chunk_idx++) {

            uint8_t already_have =
                static_cast<uint8_t>(stored_chunks.contains(chunk_idx));
            co_await asio::async_write(socket, asio::buffer(&already_have, 1),
                                       use_awaitable);
            if (already_have) {
                printf("Skipping recv of chunk %u (we already have it)\n",
                       chunk_idx);
                continue;
            }

            stored_chunks.insert(chunk_idx);

            printf(" chunk_idx: %u\n", chunk_idx);

            (void)snprintf(input_buf.data(), input_buf.size(),
                           "client_input_%u.mp4", chunk_idx);

            // receive work
            size_t nwritten = co_await DISPLAY_SPEED(
                socket_recv_file(socket, input_buf.data(), error));
            printf("Read %zu bytes from server\n", nwritten);
        }

        // Once output has been received, encode it.

        // auto vdec = DecodeContext::open(input_buf.data());
        // this should take a parameter for output filename
        // TODO remove all the hardcoding.
        // const char* outf = "client_output.mp4";
        // (void)snprintf(output_buf.data(), output_buf.size(),
        //                "client_output_%zu.mp4", work_idx);
        // main_encode_loop(output_buf.data(), std::get<DecodeContext>(vdec));

        // printf("Finished encoding '%s', output in : '%s'\n",
        // input_buf.data(),
        //        output_buf.data());

        // Send the encoded result to the server.
        // need to check
        // co_await DISPLAY_SPEED(
        //     socket_send_file(socket, output_buf.data(), error));

        fmt_buf output_buf;
        (void)snprintf(output_buf.data(), output_buf.size(), "enc_%u%u%u%u.mp4",
                       work.low_idx, work.high_idx, work.nskip, work.ndecode);

        encode_frame_range(work, output_buf.data());

        co_await DISPLAY_SPEED(
            socket_send_file(socket, output_buf.data(), error));

        // // send filler/junk data
        // uint32_t todo_delete = 4;
        // co_await asio::async_write(socket, asio::buffer(&todo_delete, 4),
        //                            use_awaitable);
        // co_await asio::async_write(socket, asio::buffer(&todo_delete, 4),
        //                            use_awaitable);
        // todo_delete = 0;
        // // write "null terminator"
        // co_await asio::async_write(socket, asio::buffer(&todo_delete, 4),
        //                            use_awaitable);

        printf("Uploaded %s to server\n", output_buf.data());
    }
    co_return;
}

// the problem with using the same buffer and holding the decoder
// is that you prevent other worker threads from using the decoder.

// Unfortunately there's really not much you can do about that.
// Because you would just have to store a huge number of frames.

// there's a memory leak with x265_malloc but what can I do about that...
// I mean actually I might be able to do something about it.
// I should double check if I had that one flag enabled for extra
// tracing info or whatever.

// tests vs av1an
//  ./target/release/av1an -i ~/avdist/OUTPUT28_29.mp4 --passes 1 --split-method
//  none --concat mkvmerge --verbose --workers 4 -o ni.mkv -x 60 -m lsmash
//  --pix-format yuv420p

// 21.23 s (divien) vs 29.16 s (av1an).

// 40.38 s (divien) vs 60.02 s (av1an)

// time ./target/release/av1an -y -i ~/avdist/test_x265.mp4 -e aom --
// passes 1 --split-method none --concat mkvmerge --verbose --workers 4 -o
// ni.mkv -x 60 -m lsmash --pix-forma t yuv420p

// 464.14 s (av1an) - 8.96 fps
// 340.91 s (divien) - 12.1 fps

// ok so ffmpeg does actually store a refcount, but only for the
// buffers themselves.
//  buf->buffer->refcount

// ok so ...
// av_frame_unref seems to free the object that holds the reference itself,
// but not the UNDERLYING buffers (I think).

// yeah ok so it really does do the refcount thing as expected.
// The underlying buffer will still live somewhere else, and it's up
// to the encoder to call unref(), I believe.
// But we still have to call unref too. I'm PRETTY SURE anyway.
// We should double check that the frames we receive from the
// decoder are in fact ref counted.

// but now we have to look into this buffer_replace() (libavutil/buffer.c)
// function.

// I'm pretty sure buffer_replace basically frees the thing if the refcount
// is 1. (OR 0). I want to double check tho.

// TODO perhaps for debug/info purposes we can compute the "overhead"
// of decoding unused frames? This would be for decoding segmented stuff.

// There should also be periodic checks to check if the connection is still
// alive. That way we aren't stuck with cpu0 encodes which fail because
// the server disconnected.

// ok next step is to do the segmenting and decoding thingy.

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>

void my_handler(int s) {
    printf("Caught signal %d\n", s);
    exit(EXIT_FAILURE);
}

// TODO handle client disconnecting,
// add that back to the chunk queue

// TODO make protocol ask client
// which chunks it's missing.
// and only send

// for verify mode/dont trust we could request
// a hash of the segments to make sure.

void run_server_full(const char* from_file) {
    printf("Preparing and segmenting video file '%s'...\n", from_file);

    unsigned int nb_segments = 0;
    // ok we need to get the packet counts out of this.
    auto work_list = server_prepare_work(from_file, nb_segments);
    // use this to iterate over segments for concatenation later on
    DvAssert(nb_segments > 0);

    // do line search to find which segments contain our stuff
    // TODO optimize it tho.
    // wait yeah this is more efficient done all at once i think
    // because of sorted property.

    // auto work_list_copy = work_list;

    ServerData data{.orig_work_size = (uint32_t)work_list.client_chunks.size(),
                    .chunks_done = 0,
                    .work = std::move(work_list),
                    .work_list_mutex = {}};

    printf("Starting server...\n");
    asio::io_context io_context(1);

    // this always counts as work to the io_context. If this is not
    // present, then .run() will automatically stop on its own
    // if we remove the socket waiting code.
    asio::signal_set signals(io_context, SIGINT, SIGTERM);
    signals.async_wait([&](auto, auto) {
        // TODO: would be nice to have graceful shutdown
        // like we keep track of how many ctrl Cs.
        // first one stops sending new chunks,
        // second one hard shuts. something like that.
        io_context.stop();

        exit(EXIT_FAILURE);
    });

    std::thread server_stopper(&server_stopper_thread, std::ref(io_context),
                               std::ref(data));

    co_spawn(io_context, run_server(io_context, from_file, data), detached);

    // I think it will block the thread.
    // yeah so this needs to be done on another thread...

    // TODO ensure this is optimized with the constructors and
    // everything

    // TODO maybe only pass the relevant data here
    // just hope that doesn't involve any extra copying

    // all the co_spawns are safe I think because everything terminates
    // at the end of this function

    // maybe I could just create another async task to check for
    // completion of tasks. That uses condition variable or something.

    io_context.run();

    server_stopper.join();

    // printf("Concatenating video segments...\n");
    // DvAssert(concat_segments_iterable(
    //              [&]<typename F>(F use_func) {
    //                  fmt_buf buf;
    //                  fmt_buf buf2;
    //                  for (auto& x : work_list_copy) {
    //                      // TODO as an optimization I could reuse the same
    //                      // buffer technically.
    //                      x.fmt_name(buf.data());
    //                      (void)snprintf(buf2.data(), buf2.size(),
    //                                     "recv_client_%s", buf.data());
    //                      use_func(buf2.data());
    //                  }
    //              },
    //              "FINAL_OUTPUT.mp4") == 0);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("DiViEn: must specify at least 2 args.\n");
        return -1;
    }

    struct sigaction sigIntHandler {};

    sigIntHandler.sa_handler = my_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;

    sigaction(SIGINT, &sigIntHandler, nullptr);

    av_log_set_callback(avlog_do_nothing);

    // ok now we have our actual TCP server/client setup here.
    // next step AFTER this is to setup async
    // so that we can handle multiple clients at once easily.
    // Or at least multithreading or whatever.
    // Honestly yeah let's just use async.
    auto mode = std::string_view(argv[1]);
    try {

        if (mode == "server") {
            if (argc < 3) {
                printf(
                    "DiViEn: must specify the input video for server mode.\n");
                return -1;
            }

            run_server_full(argv[2]);
        } else if (mode == "client") {

            // TODO deduplicate code, as it's exact same between client and
            // server
            // a really cursed way would be with both executing this code and
            // then a goto + table.
            // Like you would do if server, set index=0, goto this. code,
            // goto[index]. same for other one
            asio::io_context io_context(1);

            asio::signal_set signals(io_context, SIGINT, SIGTERM);
            signals.async_wait([&](auto, auto) {
                io_context.stop();

                // TODO access exit function safely if possible
                // not sure how thread safe this is
                exit(EXIT_FAILURE);
            });

            tcp::resolver resolver(io_context);

            auto endpoints = resolver.resolve("localhost", "7878");
            asio::error_code error;

            tcp_socket socket(io_context);
            asio::connect(socket, endpoints, error);

            if (error) {
                // TODO any way to avoid constructing std::string?
                auto msg = error.message();
                printf("Error occurred connecting to server: %s\n",
                       msg.c_str());
                return -1;
            }

            co_spawn(io_context,
                     run_client(io_context, std::move(socket), error),
                     detached);

            io_context.run();

        } else if (mode == "standalone") {
            if (argc < 3) {
                printf(
                    "Insufficient arguments specified for standalone mode.\n");
                return -1;
            }

            auto vdec = DecodeContext::open(argv[2]);

            main_encode_loop("standalone_output.mp4",
                             std::get<DecodeContext>(vdec));
        } else {
            printf("unknown mode '%.*s'.\n", (int)mode.size(), mode.data());
        }

    } catch (std::exception& e) {
        printf("Exception: %s\n", e.what());
    }
}

// bruh what's a solution...
// For this

// I mean for distributed we will know the packet count ahead of time.
// so that could help or whatever. Plus we will know the offsets.
// Hmm...
// This is a complicated problem.
// Actuallly for chunked we will HAVE to run multiple decoders anyway,
// we have no choice.
//      - when we do get to this, an optimization would be to separate
//        the muxer and the decoder so we don't have to allocate another
//        decoder to decode packets from the next stream.

// I mean, for the purposes of distributed we can ignore the problem for now.
// But one idea could be like... to have a buffer full of frames that keeps
// decoding totally independently of workers, just fills up a buffer of free
// frames. And then the workers get assigned ranges of the buffer.
// In general this sounds pretty complicated though.

// Well perhaps we could use the make_frame_writable thing or whatever.

// Well one idea is we could

// io_uring, IOCP. Can we make ASIO use these?

int main_unused(int argc, char* argv[]) {
    if (signal(SIGSEGV, segvHandler) == SIG_ERR) {
        w_err("signal(): failed to set SIGSEGV signal handler\n");
    }

    if (argc != 2) {
        w_err("DiViEn: invalid number of arguments\n"
              "   usage: DiViEn  <video_file>\n");
        return -1;
    }
    const char* url = argv[1];

    // I think maybe next step will be cleaning up the code. So that it will
    // be feasible to continue working on this.

    // TODO maybe add option to not copy timestamps.
    // Dang this stuff is complicated.

    // 1432 max index

    // ok so unfortunately it does seem like it's possible to
    // have totally adjacent segments with broken framecounts.
    // In which case I believe we need to concatenate longer
    // segments.
    // But hopefully it isn't possible that the very first segment
    // has extra packets.

    return 0;

    // av_log_set_level(AV_LOG_VERBOSE);

    // segment_video("/home/yusuf/avdist/test_x265.mp4", "OUTPUT%d.mp4");
    // so concat code works perfectly fine
    // unsigned int nb_segments = 30;

    // TODO: Dynamically count number of segments made.
    // Is there a way to do that from ffmpeg directly?
    // identify_broken_segments(nb_segments);

    // can assume argv[1] is now available

    // so as soon as you use something like std::cout, the binary size
    // increases greatly...

    // so we should probably find a way to not use things that increase the
    // binary size a lot...

    // bro WHY are there dropped frames...
    // when decoding man... I just don't get all of them.

    // TODO allow setting log level via CLI
    av_log_set_callback(avlog_do_nothing);
    // should I just modify the ffmpeg library to do what
    // I need it to do? Hmm... Well it won't work with the system
    // version of the library then, unfortunately.
    // av_log_default_callback();

    // TODO move all this to another function

    auto vdec = DecodeContext::open(url);

    // TODO use std::expected instead?
    std::visit(
        [](auto&& arg) {
            using T = std::decay_t<decltype(arg)>;

            if constexpr (std::is_same_v<T, DecodeContext>) {
                main_encode_loop("output.mp4", arg);

            } else if constexpr (std::is_same_v<T, DecoderCreationError>) {
                auto error = arg;

                // TODO move all this into its own function
                if (error.type == DecoderCreationError::AVError) {
                    std::array<char, AV_ERROR_MAX_STRING_SIZE> errbuf{};
                    av_make_error_string(errbuf.data(), errbuf.size(),
                                         error.averror);

                    (void)fprintf(stderr, "Failed to initialize decoder: %s\n",
                                  errbuf.data());

                } else {
                    std::string_view errmsg = error.errmsg();
                    (void)fprintf(stderr,
                                  "Failed to initialize decoder: %.*s\n",
                                  (int)errmsg.size(), errmsg.data());
                }
            } else {
                static_assert(always_false_v<T>);
            }
        },
        vdec);

    return 0;
}
