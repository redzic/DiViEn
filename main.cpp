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
#include <mutex>
#include <pthread.h>
#include <span>
#include <thread>
#include <unistd.h>
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

// #define ERASE_LINE_ANSI "\x1B[1A\x1B[2K" // NOLINT
#define ERASE_LINE_ANSI "" // NOLINT

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

template <class> inline constexpr bool always_false_v = false;

template <class result_t = std::chrono::milliseconds,
          class clock_t = std::chrono::steady_clock,
          class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const& start) {
    return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

template <class result_t = std::chrono::milliseconds,
          class clock_t = std::chrono::steady_clock,
          class duration_t = std::chrono::milliseconds>
auto dist_ms(std::chrono::time_point<clock_t, duration_t> const& start,
             std::chrono::time_point<clock_t, duration_t> const& end) {
    return std::chrono::duration_cast<result_t>(end - start);
}

auto now() { return std::chrono::steady_clock::now(); }

std::mutex global_decoder_mutex;  // NOLINT
unsigned int global_chunk_id = 0; // NOLINT

// for printing progress
std::condition_variable cv; // NOLINT
// is mutex fast?
// is memcpy optimized on windows? Memchr sure isn't.
// is there a "runtime" that provides optimized implementations?
// can I override the standard memcpy/memchr functions?
std::mutex cv_m; // NOLINT

std::atomic<uint32_t> nb_frames_done(0);  // NOLINT
std::atomic<uint32_t> nb_threads_done(0); // NOLINT

bool all_workers_finished() { return nb_threads_done == NUM_WORKERS; }

// If pkt is refcounted, we shouldn't have to copy any data.
// But the encoder may or may not create a reference.
// I think it probably does? Idk.
int encode(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt,
           FILE* ostream) {
    // frame can be null, which is considered a flush frame
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

// allocates entire buffer upfront
// single threaded version
int encode_frames(const char* file_name, std::span<AVFrame*> frame_buffer) {
    DvAssert(!frame_buffer.empty());

    // TODO is it possible to reuse encoder instances?
    const auto* codec = avcodec_find_encoder_by_name("libaom-av1");
    // const auto* codec = avcodec_find_encoder_by_name("libx264");
    // so avcodeccontext is used for both encoding and decoding...
    // TODO make this its own class
    auto* avcc = avcodec_alloc_context3(codec);
    avcc->thread_count = THREADS_PER_WORKER;

    if (avcc == nullptr) {
        w_err("failed to allocate encoder context\n");
        return -1;
    }

    auto pkt = make_resource<AVPacket, av_packet_alloc, av_packet_free>();

    avcc->width = frame_buffer[0]->width;
    avcc->height = frame_buffer[0]->height;
    avcc->time_base = (AVRational){1, 25};
    avcc->framerate = (AVRational){25, 1};

    avcc->pix_fmt =
        av_pix_fmt_supported_version((AVPixelFormat)frame_buffer[0]->format);

    // av_opt_set(avcc->priv_data, "crf", "25", 0);
    // av_opt_set(avcc->priv_data, "preset", "veryfast", 0);
    // AOM:
    av_opt_set(avcc->priv_data, "cpu-used", "6", 0);
    av_opt_set(avcc->priv_data, "end-usage", "q", 0);
    av_opt_set(avcc->priv_data, "enable-qm", "1", 0);
    av_opt_set(avcc->priv_data, "cq-level", "18", 0);

    // RAV1E:
    // av_opt_set(avcc->priv_data, "speed", "7", 0);

    int ret = avcodec_open2(avcc, codec, nullptr);
    if (ret < 0) {
        w_err("failed to open codec\n");
        return ret;
    }

    // C-style IO is needed for binary size to not explode on Windows with
    // static linking

    // TODO use unique_ptr as wrapper resource manager
    make_file(file, file_name, "wb");

    for (auto* frame : frame_buffer) {
        // required
        frame->pict_type = AV_PICTURE_TYPE_NONE;
        encode(avcc, frame, pkt.get(), file.get());
        // actually nvm I'm not sure why this data seems wrong.
        nb_frames_done++;
    }
    // need to send flush packet
    encode(avcc, nullptr, pkt.get(), file.get());

    avcodec_free_context(&avcc);

    return 0;
}

int encode_chunk(unsigned int chunk_idx, std::span<AVFrame*> framebuf) {
    std::array<char, 64> buf{};

    // this should write null terminator
    (void)snprintf(buf.data(), buf.size(), "file %d.mp4", chunk_idx);

    // return encode_frames(buf.data(), framebuf, n_frames);
    return encode_frames(buf.data(), framebuf);
}

// framebuf is start of frame buffer that worker can use
int worker_thread(unsigned int worker_id, DecodeContext& decoder) {
    while (true) {
        // should only access decoder once lock has been acquired
        global_decoder_mutex.lock();

        // decode CHUNK_FRAME_SIZE frames into frame buffer
        int frames = run_decoder(decoder, worker_id * CHUNK_FRAME_SIZE,
                                 CHUNK_FRAME_SIZE);

        // error decoding
        if (frames <= 0) {
            nb_threads_done++;
            cv.notify_one();
            global_decoder_mutex.unlock();
            return frames;
        }

        // these accesses are behind mutex so we're all good
        auto chunk_idx = global_chunk_id++;
        // increment for next chunk
        // TODO delete line global_chunk_id++;

        // can assume frames are available, so unlock the mutex so
        // other threads can use the decoder
        global_decoder_mutex.unlock();

        // a little sketchy but in theory this should be fine
        // since framebuf is never modified
        int ret = encode_chunk(chunk_idx, {decoder.framebuf.data() +
                                               (worker_id * CHUNK_FRAME_SIZE),
                                           (size_t)frames});

        if (ret != 0) {
            // in theory... this shouldn't need to happen as this is an encoding
            // error
            // mutex was already unlocked so we don't unlock.

            nb_threads_done++;

            // in normal circumstances we return from infinite loop via decoding
            // error (which we expect to be EOF).
            cv.notify_one();

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

// decodes everything
// TODO need to make the output of this compatible with libav
// logging stuff. Maybe I can do that with a callback
// so that I can really handle it properly.
// This relies on global state. Do not call this function
// more than once.
void main_encode_loop(const char* out_filename, DecodeContext& d_ctx) {
    // uh... we really need a safe way to do this and stuff
    // this is really bad... just assuming enough frames are
    // available

    auto start = now();

    // TODO use stack array here
    // even if we have dynamic workers, I believe we
    // can use alloca. Not really totally ideal but,
    // whatever. Or just array with max capacity.
    std::vector<std::thread> thread_vector{};
    thread_vector.reserve(NUM_WORKERS);

    // spawn worker threads
    for (unsigned int i = 0; i < NUM_WORKERS; i++) {
        thread_vector.emplace_back(&worker_thread, i, std::ref(d_ctx));
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
        std::unique_lock<std::mutex> lk(cv_m);

        // so for some reason this notify system isn't good enough
        // for checking if all threads have completed.
        auto local_start = now();
        // TODO Is this guaranteed to deal with "spurious wakes"?
        auto status = cv.wait_for(lk, std::chrono::seconds(1));

        auto n_frames = nb_frames_done.load();
        auto frame_diff = n_frames - last_frames;
        last_frames = n_frames;
        // since we waited exactly 1 second, frame_diff is the fps,
        // unless the status is no_timeout.

        auto local_now = now();
        auto total_elapsed_ms = dist_ms(start, local_now).count();

        // So this part of the code can actually run multiple times.
        // For each thread that signals completion.
        // Well this does work for avoiding extra waiting unnecessarily.
        // TODO simplify/optimize this code if possible

        if (status == std::cv_status::no_timeout) [[unlikely]] {
            // this means we didn't wait for the full time
            auto elapsed_local_ms = dist_ms(local_start, local_now).count();
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
        printf(ERASE_LINE_ANSI "frame= %d  (%s fps curr, %s fps avg)\n",
               n_frames, local_fps_fmt.data(), avg_fps_fmt.data());

        static_assert(NUM_WORKERS >= 1);

        if (all_workers_finished()) {
            break;
        }
    }

    // In theory all the threads have already exited here
    // but we need to call .join() anyways.
    for (auto& t : thread_vector) {
        t.join();
    }

    // there is no active lock on the mutex since all threads
    // terminated, so global_chunk_id can be safely accessed.
    raw_concat_files(out_filename, global_chunk_id, true);
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

#include <asio.hpp>
#include <asio/co_spawn.hpp>
#include <asio/detached.hpp>
#include <asio/io_context.hpp>
#include <asio/ip/tcp.hpp>
#include <asio/signal_set.hpp>
#include <asio/write.hpp>
#include <cstdio>
#include <iostream>

using asio::ip::tcp;

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

constexpr size_t TCP_BUFFER_SIZE = 2048z * 32; // 64 Kb buffer

// TODO move networking code to another flie.

// Returns number of bytes read.
size_t
socket_write_all_from_file(tcp::socket& socket,
                           const char* filename) { // Open a file in read mode
    make_file(fptr, filename, "rb");

    std::array<uint8_t, TCP_BUFFER_SIZE> read_buf;
    uint32_t header = 0;
    // TODO stop ignoring errors
    asio::error_code error;

    // if return value
    // TODO error handling
    size_t n_read = 0;
    size_t total_read = 0;
    while ((n_read = fread(read_buf.data(), 1, read_buf.size(), fptr.get())) >
           0) {
        // I believe this should work.
        // TODO FIX ENDIANNESS!!!!
        header = n_read;
        asio::write(socket, asio::buffer(&header, sizeof(header)), error);
        asio::write(socket, asio::buffer(read_buf, n_read), error);
        total_read += n_read;
    }
    header = 0;
    DvAssert(
        asio::write(socket, asio::buffer(&header, sizeof(header)), error) == 4);

    return total_read;
}

// TODO check on godbolt if the compiler auto dedups
// these calls, or if it inlines them or what.
// I imagine it prob doesn't. But who knows. I mean .
// Yeah probably not. But we should check
// how to make it dedup it.
template <typename Functor> size_t print_transfer_speed(Functor f) {
    auto start = now();
    size_t bytes = f();

    auto total_elapsed_ms = dist_ms(start, now()).count();
    auto mb_s = static_cast<double>(bytes) /
                static_cast<double>(1000 * total_elapsed_ms);

    // TODO : to make this more accurate, only count
    // the times we are actually waiting on the file (not disk write
    // times)
    printf(" [%ld ms] %.1f MB/s throughput (%.0f Mbps)\n", total_elapsed_ms,
           mb_s, 8.0 * mb_s);
    return bytes;
}

#define DISPLAY_SPEED(arguments)                                               \
    print_transfer_speed([&]() { return arguments; })

// Returns number of bytes received
size_t socket_read_all_to_file(tcp::socket& socket, const char* filename) {
    make_file(fptr, filename, "wb");
    size_t written = 0;
    uint32_t header = 0;
    while (true) {
        // TODO figure out optimal buffer size
        std::array<uint8_t, TCP_BUFFER_SIZE> buf;
        asio::error_code error;

        // it seems like the issue is that once the file has been received,
        // this doesn't sufficiently block for the data to be receieved from the
        // client
        // we should definitely like, handle connection closes properly.
        DvAssert(asio::read(socket, asio::buffer(&header, sizeof(header)),
                            error) == 4);

        if (header == 0) {
            break;
        }

        size_t len = 0;
        DvAssert((len = asio::read(socket, asio::buffer(buf, header), error)) ==
                 header);

        // So the problem is that we rely on the connection being closed,
        // but if we need to send back data to the server, the connection will
        // not be closed.
        if (error == asio::error::eof) {
            printf("UNEXPECTED EOF\n");
            break; // Connection closed cleanly by peer (server).
        } else if (error) {
            throw asio::system_error(error); // Some other error.}
        }

        // assume successful call (in release mode)
        DvAssert(fwrite(buf.data(), 1, len, fptr.get()) == len);
        written += len;
        // printf("Wrote %zu bytes to %s\n", written, filename);
    }
    // printf("Wrote %zu bytes to %s\n", written, filename);

    return written;
}

// new protocol, receive and send N chunks
// When server closes connection, client closes also.
// I Guess we don't have to actually change anything.

// source_file is the file to segment and encode
void run_server(const char* source_file) {
    // we need a list of file names here
    // Ideally we should return some data structure
    // and generate the filename with snprintf
    // auto seg_result = segment_video_fully(source_file);

    asio::io_context io_context;
    asio::error_code error;

    tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 7878));

    printf("Listening for connections (synchronous)...\n");
    int conn_num = 1;
    while (true) {
        tcp::socket socket(io_context);
        acceptor.accept(socket);

        printf("[TCP] Connection %d accepted\n", conn_num);

        // I HIGHLY doubt this is not inlined lol.
        // I'm surprised you can pass lambdas that take captures to
        // templates like this though, that's actually insane.

        // TODO (long term) Honestly using bittorrent to distribute the chunks
        // among other stuff is probably the way to go.

        // Send payload to encode to client
        auto bytes1 =
            DISPLAY_SPEED(socket_write_all_from_file(socket, source_file));
        printf("Sent %zu bytes to client\n", bytes1);

        // TODO. It would be faster to transfer the encoded packets
        // as they are complete, so we do stuff in parallel.
        // Instead of waiting around for chunks to be entirely finished.

        // Receive back encoded data
        // TODO the display on this is totally misleading
        // because it takes into account the encoding time as well.
        // Fixing this would require a redesign to the protocol I guess.
        auto bytes =
            DISPLAY_SPEED(socket_read_all_to_file(socket, "client.mp4"));
        printf("Read back %zu bytes from client\n", bytes);

        // read all data

        printf("[TCP] Connection %d closed\n", conn_num++);
    }
}

void run_client() {
    asio::io_context io_context;
    tcp::resolver resolver(io_context);

    auto endpoints = resolver.resolve("localhost", "7878");

    tcp::socket socket(io_context);
    asio::connect(socket, endpoints);

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

    // where to dump the input file from the server
    const char* input_url = "input.mp4";

    // Read payload from server.
    DISPLAY_SPEED(socket_read_all_to_file(socket, input_url));

    // Once output has been received, encode it.

    auto vdec = DecodeContext::open(input_url);
    // this should take a parameter for output filename
    // TODO remove all the hardcoding.
    const char* outf = "output.mp4";
    main_encode_loop(outf, std::get<DecodeContext>(vdec));

    printf("Finished encoding '%s', output in : '%s'\n", input_url, outf);

    // Send the encoded result to the server.
    DISPLAY_SPEED(socket_write_all_from_file(socket, outf));
    printf("Uploaded %s to server\n", outf);
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Must specify 2 args.\n");
        return -1;
    }

    // av_log_set_callback(avlog_do_nothing);

    const char* from_file = "/home/yusuf/avdist/test_x265.mp4";
    // const char* from_file = "/home/yusuf/avdist/OUTPUT0.mp4";

    // ok now we have our actual TCP server/client setup here.
    // next step AFTER this is to setup async
    // so that we can handle multiple clients at once easily.
    // Or at least multithreading or whatever.
    // Honestly yeah let's just use async.
    auto mode = std::string_view(argv[1]);
    try {

        if (mode == "server") {
            run_server(from_file);
        } else if (mode == "client") {
            run_client();
        } else {
            printf("unknown mode '%.*s'.\n", (int)mode.size(), mode.data());
        }

    } catch (std::exception& e) {
        printf("Exception: %s\n", e.what());
    }
    return 0;
}
// TODO I'm pretty sure the progress bar is totally wrong.
// It's counting frames sent instead of packets received.

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

                // int frames = decode_loop(arg);
                // printf("decoded %d frames\n", frames);

                // DvAssert(run_decoder(arg, 0, 60) == 60);
                // printf("initial decode of 60 frames succeeded\n");

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
