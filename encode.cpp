#include "encode.h"

#include "common.h"
#include "decode.h"
#include "ffutil.h"
#include "progress.h"
#include "resource.h"
#include "timing.h"
#include "util.h"
#include <cstring>
#include <libavutil/frame.h>

namespace fs = std::filesystem;

#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>

#define ERASE_LINE_ANSI "\x1B[1A\x1B[2K"

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

// TODO proper error handling, return std::expected
// caller needs to ensure they only call this once
// The e_opts should start with a '-'.
void EncoderContext::initialize_codec(AVFrame* frame, unsigned int n_threads,
                                      EncoderOpts e_opts) {
    const auto* codec = avcodec_find_encoder_by_name(e_opts.encoder_name);
    avcc = avcodec_alloc_context3(codec);
    DvAssert(avcc);
    pkt = av_packet_alloc();
    DvAssert(pkt);
    avcc->thread_count = n_threads;
    // arbitrary values
    avcc->time_base = (AVRational){1, 25};
    avcc->framerate = (AVRational){25, 1};

    DvAssert(frame->width > 0);
    DvAssert(frame->height > 0);

    avcc->width = frame->width;
    avcc->height = frame->height;
    avcc->pix_fmt = av_pix_fmt_supported_version((AVPixelFormat)frame->format);

    for (size_t i = 0; i < e_opts.n_param_pairs; i++) {
        // TODO print error message for failed param
        // TODO: according to godbolt, is += 2 better in the loop condition?
        const char* key = e_opts.params[2 * i];
        const char* value = e_opts.params[2 * i + 1];
        DvAssert(strlen(key) >= 1);
        DvAssert(strlen(value) >= 1);
        DvAssert(key[0] == '-');
        int ret = av_opt_set(avcc->priv_data, &key[1], value, 0);
        DvAssert(ret == 0);
        const char* err = nullptr;
        // TODO: come up with mechanism to integrate the progress bar
        // printing, so the two don't conflict.
        if (ret == AVERROR_OPTION_NOT_FOUND) {
            err = "option not found";
        } else if (ret == AVERROR(ERANGE)) {
            err = "value out of range";
        } else if (ret == AVERROR(EINVAL)) {
            err = "invalid value";
        } else {
            err = "unspecified error";
        }
        if (ret) {
            (void)fprintf(stderr, "\n\nWARNING: Failed to set %s=%s: %s\n\n",
                          key, value, err);
        }
    }

    int ret = avcodec_open2(avcc, codec, nullptr);
    DvAssert(ret == 0 && "Failed to open encoder codec");
}

// If pkt is refcounted, we shouldn't have to copy any data.
// But the encoder may or may not create a reference.
// I think it probably does? Idk.
int encode_frame(AVCodecContext* enc_ctx, AVFrame* frame, AVPacket* pkt,
                 FILE* ostream, std::atomic<uint32_t>& frame_count) {
    // frame can be null, which is considered a flush frame
    DvAssert(enc_ctx);
    if (frame) {
        DvAssert(avframe_has_buffer(frame));
        DvAssert(frame->pict_type == AV_PICTURE_TYPE_NONE);
    }

    int ret = avcodec_send_frame(enc_ctx, frame);

    if (ret < 0) {
        // TODO deduplicate, make macro or function to do this
        auto errmsg = av_strerr(ret);
        if (!frame) {
            printf("error sending flush frame to encoder: %s\n", errmsg.data());
        } else {
            printf("error sending frame to encoder: %s\n", errmsg.data());
        }

        return ret;
    }

    while (true) {
        int ret = avcodec_receive_packet(enc_ctx, pkt);
        // why check for eof though?
        // what does eof mean here?
        // actually this doesn't really seem correct
        // well we are running multiple encoders aren't we?
        // so that's why we get eof. I guess. idk.
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            return 0;
        } else if (ret < 0) {
            printf("unspecified error during encoding\n");
            return ret;
        }
        frame_count++;

        // can write the compressed packet to the bitstream now

        // TODO make this non-blocking with ASIO
        DvAssert(fwrite(pkt->data, 1, pkt->size, ostream) == (size_t)pkt->size);

        av_packet_unref(pkt);
    }

    return 0;
}

int encode_frames(const char* file_name, std::span<AVFrame*> framebuf,
                  EncodeLoopState& state, unsigned int n_threads,
                  EncoderOpts e_opts) {
    DvAssert(!framebuf.empty());

    EncoderContext encoder;
    encoder.initialize_codec(framebuf[0], n_threads, e_opts);

    // C-style IO is needed for binary size to not explode on Windows with
    // static linking

    // TODO use unique_ptr as wrapper resource manager
    make_file(file, file_name, "wb");

    int64_t pts = 0;
    for (auto* frame : framebuf) {
        // required
        frame->pict_type = AV_PICTURE_TYPE_NONE;
        frame->pts = pts++;
        encode_frame(encoder.avcc, frame, encoder.pkt, file.get(),
                     state.nb_frames_done);
    }
    // need to send flush packet after we're done
    encode_frame(encoder.avcc, nullptr, encoder.pkt, file.get(),
                 state.nb_frames_done);

    return 0;
}

static void get_frame_defaults(AVFrame* frame) {
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

// Somewhere here we leak memory
// TODO add asserts for decoder handling
void encode_frame_range(FrameAccurateWorkItem& data, const char* ofname) {
    // we only skip frames on the first chunk, otherwise it wouldn't
    // make any sense. All chunks have frames we need to decode.
    EncoderContext encoder;
    // now we have to encode exactly ndecode frames
    auto nleft = data.ndecode;

    make_file(ofptr, ofname, "wb");

    bool enc_was_init = false;
    std::atomic<uint32_t> nframes_done = 0;

    printf("frame= 0\n");

    AVFrame frame;
    get_frame_defaults(&frame);

    int64_t fixed_pts = 0;

    for (uint32_t idx = data.low_idx; idx <= data.high_idx; idx++) {

        std::array<char, 128> input_fname;
        (void)snprintf(input_fname.data(), input_fname.size(),
                       "client_input_%d.mp4", idx);

        printf("=========== NEW SUBCHUNK. OPENING %s FOR READING.\n",
               input_fname.data());

        auto dres = DecodeContext::open(input_fname.data(), 0);
        auto& dc = std::get<DecodeContext>(dres);

        // perhaps we could initialize all these decoders at the same time...
        // to save time.
        // TODO reuse underlying frame buffer

        // TODO split loop if possible
        if (idx == data.low_idx) {
            // means we first need to decode nskip frames
            // initially, frames are not writable. Because they
            // have
            // not been properly allocated yet, their actual
            // buffers
            // I mean. The decoder allocates those buffers.
            for (uint32_t nf = 0; nf < data.nskip; nf++) {

                // This can fail if you run 2 clients in the same directory.
                // TODO prevent that.
                DvAssert(decode_next(dc, &frame) == 0);
                if (nf == 0) {
                    // TODO use proper amount of threads
                    encoder.initialize_codec(&frame, 8, AOMENC);
                    enc_was_init = true;
                }
            }
        } else {
            DvAssert(enc_was_init);
        }

        // TODO allow configurable frame size when constructing
        // decoder, to avoid wasting memory
        while (nleft > 0) {
            printf("%d frames left\n\n", nleft);

            int ret = 0;
            // if (avframe_has_buffer(&frame))
            //     DvAssert(av_frame_make_writable(&frame) == 0);

            if ((ret = decode_next(dc, &frame)) != 0) {
                // DvAssert();
                break;
            }

            if (!enc_was_init) {
                // can happen if nskip is 0
                DvAssert(data.nskip == 0);
                DvAssert(idx == data.low_idx);
                encoder.initialize_codec(&frame, 8, AOMENC);
                enc_was_init = true;
            }

            // TODO find out of intel TBB can override posix thread creation
            // function/std::thread

            DvAssert(avframe_has_buffer(&frame));
            DvAssert(encoder.pkt);

            // fixing pts is required, otherwise encoder can get confused
            // when it sees the next packet. And weird errors will happen.
            // TODO debug assert that original pts are in order.
            frame.pts = fixed_pts++;
            frame.pict_type = AV_PICTURE_TYPE_NONE;
            DvAssert(encode_frame(encoder.avcc, &frame, encoder.pkt,
                                  ofptr.get(), nframes_done) == 0);
            printf("frame= %u\n", nframes_done.load());

            nleft--;
        }
    }
    av_frame_unref(&frame);

    DvAssert(ofptr != nullptr);
    printf("Sending flush packet...\n");
    // why does this return an error?
    encode_frame(encoder.avcc, nullptr, encoder.pkt, ofptr.get(), nframes_done);
    printf("Chunk finished [%d frames], output = '%s'\n", nframes_done.load(),
           ofname);
}

// TODO: make output filename configurable
// the passed folder_name should include a slash
// folder_name and prefix should be string_view.
#define chunk_fname(buf_var, folder_name, prefix, chunk_idx)                   \
                                                                               \
    char buf_var[(folder_name).size() + (prefix).size() + 64];                 \
    (void)snprintf((char*)(buf_var), sizeof(buf_var), SVF SVF "_chunk_%u.mp4", \
                   SV(folder_name), SV(prefix), chunk_idx);

AlwaysInline int encode_chunk(std::string_view base_path,
                              std::string_view prefix, unsigned int chunk_idx,
                              std::span<AVFrame*> framebuf,
                              EncodeLoopState& state, unsigned int n_threads,
                              EncoderOpts e_opts, ChunkData frange) {
    chunk_fname(buf, base_path, prefix, chunk_idx);
    int ret = encode_frames((char*)buf, framebuf, state, n_threads, e_opts);
    dump_chunk(state.resume_m, state.resume_data, state.p_fname, frange);
    return ret;
}

// framebuf is start of frame buffer that worker can use
// TODO pass n_threads and chunk size in with some shared state
int worker_thread(std::string_view base_path, std::string_view prefix,
                  unsigned int worker_id, DecodeContext& decoder,
                  EncodeLoopState& state, EncoderOpts e_opts) {
    for (;;) {
        // for (size_t i = 0; i < state.chunk_frame_size; i++) {
        //     size_t idx = (size_t)worker_id * state.chunk_frame_size + i;
        // if (avframe_has_buffer(decoder.framebuf[idx])) {
        //     DvAssert(av_frame_make_writable(decoder.framebuf[idx]) == 0);
        // }
        // }

        // should only access decoder once lock has been acquired
        // uh should we replace with like unique_lock or lock_guard
        // or something like that?
        // idk how save this is
        state.global_decoder_mutex.lock();

        auto chunk_idx = state.global_chunk_id++;
        int frames = 0;

        // get chunk
        // TODO: fix locking of mutex that can happen if
        // resume file contains invalid data.
        for (;;) {
            auto should_skip = state.resume_data.contains(chunk_idx);
            // decode CHUNK_FRAME_SIZE frames into frame buffer
            // decode into same frame to thrash memory less
            int frames2 = 0;
            if (should_skip) {
                // TODO reuse loop counter variable for frames2
                for (unsigned int n = state.chunk_frame_size; n; n--) {
                    frames = run_decoder(
                        decoder, (size_t)worker_id * state.chunk_frame_size, 1);
                    if (frames <= 0) {
                        // TODO handle error by releasing mutex
                        DvAssert(frames2 > 0);
                        break;
                    }
                    frames2 += frames;
                }
                DbgDvAssert(frames2 == (state.resume_data[chunk_idx].high -
                                        state.resume_data[chunk_idx].low + 1));
                state.nb_frames_skipped += frames2;

                chunk_idx = state.global_chunk_id++;
                continue;
            }

            // I think something with chunk_idx incrementing is totally
            // wrong...
            frames =
                run_decoder(decoder, (size_t)worker_id * state.chunk_frame_size,
                            state.chunk_frame_size);
            // eof or decoding error
            if (frames <= 0) {
                state.nb_threads_done++;
                state.cv.notify_one();
                state.global_decoder_mutex.unlock();
                return frames;
            }
            break;

            // end of loop
        }

        DvAssert(frames != 0);

        // we decoded our chunk. If that chunk should be skipped, then we skip
        // it.

        // these accesses are behind mutex so we're all good
        // increment for next chunk

        // can assume frames are available, so unlock the mutex so
        // other threads can use the decoder
        state.global_decoder_mutex.unlock();

        DvAssert(state.chunk_frame_size != 0);
        ChunkData frange = {.idx = chunk_idx,
                            .low = chunk_idx * state.chunk_frame_size,
                            .high = chunk_idx * state.chunk_frame_size +
                                    frames - 1};
        // a little sketchy but in theory this should be fine
        // since framebuf is never modified
        int ret = encode_chunk(
            base_path, prefix, chunk_idx,
            {decoder.framebuf.data() +
                 ((size_t)worker_id * (size_t)state.chunk_frame_size),
             (size_t)frames},
            state, state.n_threads, e_opts, frange);

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

void raw_concat_files(std::string_view base_path, std::string_view prefix,
                      const char* out_filename, unsigned int num_files,
                      bool delete_after) {
    std::ofstream dst(out_filename, std::ios::binary);

    for (unsigned int i = 0; i < num_files; i++) {
        chunk_fname(buf, base_path, prefix, i);
        std::ifstream src((char*)buf, std::ios::binary);
        dst << src.rdbuf();
        src.close();
        // delete file after done
        if (delete_after) {
            DvAssert(std::remove(buf) == 0);
        }
    }

    dst.close();
}

static double compute_fps(uint32_t n_frames, int64_t time_ms) {
    if (time_ms <= 0) [[unlikely]] {
        return INFINITY;
    } else [[likely]] {
        return static_cast<double>(1000 * n_frames) /
               static_cast<double>(time_ms);
    }
};

static void print_progress_internal(auto start, uint32_t n_done,
                                    uint32_t n_skipped) {
    std::array<char, 32> avg_fps_fmt;

    auto local_now = now();
    auto total_elapsed_ms = dist_ms(start, local_now);

    // So this part of the code can actually run multiple times.
    // For each thread that signals completion.
    // Well this does work for avoiding extra waiting unnecessarily.
    // TODO simplify/optimize this code if possible

    // average fps from start of encoding process
    // TODO can we convert to faster loop with like boolean flag + function
    // pointer or something? It probably won't actually end up being faster
    // due to overhead tho.
    auto avg_fps = compute_fps(n_done, total_elapsed_ms);
    // using 9.5 to align with rounding behavior of printf
    (void)snprintf(avg_fps_fmt.data(), avg_fps_fmt.size(),
                   avg_fps < 9.5 ? "%.1f" : "%0.f", avg_fps);

    // print progress
    // TODO I guess this should detect if we are outputting to a
    // terminal/pipe and don't print ERASE_LINE_ANSI if not a tty.
    if (n_skipped) {
        printf(ERASE_LINE_ANSI "frame= %d  [%s fps]   (%d skipped)\n",
               n_done + n_skipped, avg_fps_fmt.data(), n_skipped);
    } else {
        printf(ERASE_LINE_ANSI "frame= %d  [%s fps]\n", n_done,
               avg_fps_fmt.data());
    }
}

[[nodiscard]] int
chunked_encode_loop(EncoderOpts e_opts, const char* in_filename,
                    const char* out_filename, DecodeContext& d_ctx,
                    unsigned int num_workers, unsigned int chunk_frame_size,
                    unsigned int n_threads) {
    // TODO avoid dynamic memory allocation here
    auto base_path =
        fs::path(in_filename).filename().replace_extension().generic_string();

    // TODO technically could avoid copy here by using same pointer
    auto prefix = base_path;

    // TODO avoid copies and memory allocation here
    // TODO check whether output file already exists
    base_path.append("_divien/");
    std::error_code fs_ec;
    bool was_created = fs::create_directory(base_path, fs_ec);
    if (fs_ec) {
        printf(DIVIEN ": Error creating temporary directory '%s': %s\n",
               base_path.c_str(), fs_ec.message().c_str());
        return -1;
    }
    // TODO: add CLI option to abort when directory already exists. Or perhaps
    // by default it aborts and add option to reuse/overwrite same directory.

    // TODO remove copy
    auto progress_file = base_path;
    progress_file.append("progress.dvn");

    chunk_hmap resume_data;

    if (!was_created) {
        printf(DIVIEN ": Warning: using existing temporary directory '%s'\n",
               base_path.c_str());
        if (fs::exists(progress_file)) {
            // TODO maybe just change file extension to .bin
            printf(DIVIEN ": Found progress.dvn file -- Resuming mode set\n");
            resume_data = parse_chunks(progress_file.c_str());
            printf(DIVIEN
                   ": progress.dvn parsed -- found %zu complete chunks\n",
                   resume_data.size());
        }
    } else {
        printf("Using temporary directory: '%s'\n", base_path.c_str());
    }

    printf("Writing encoded output to '%s'\n", out_filename);

    EncodeLoopState state(progress_file.c_str(), num_workers, chunk_frame_size,
                          n_threads, resume_data);

    auto start = now();

    // so... basically we skip chunk indexes that are not relevant.
    // we always need to get the chunk index and hold the decoder

    // so I think for thread affinity to work we will have to spawn different
    // processes instead of threads. And somehow share memory between them.
    // TODO use VLA and placement new?
    std::vector<std::thread> thread_vector{};
    thread_vector.reserve(state.num_workers);

    // spawn worker threads
    for (unsigned int i = 0; i < state.num_workers; i++) {
        thread_vector.emplace_back(&worker_thread, base_path, prefix, i,
                                   std::ref(d_ctx), std::ref(state), e_opts);
    }

    printf("frame= 0\n");

    while (true) {
        // acquire lock on mutex I guess?
        // TODO: see if we can release this lock earlier.
        std::unique_lock<std::mutex> lk(state.cv_m);

        state.cv.wait_for(lk, std::chrono::seconds(1));

        print_progress_internal(start, state.nb_frames_done,
                                state.nb_frames_skipped);

        if (state.all_workers_finished()) {
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
    // raw_concat_files(out_filename, global_chunk_id, true);
    // TODO why does deleting files fail sometimes?
    // TODO error handling
    raw_concat_files(base_path, prefix, out_filename, state.global_chunk_id,
                     false);
    return 0;
}
