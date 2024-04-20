#include <cerrno>
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <pthread.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <variant>

#include "common.h"
#include "decode.h"
#include "encode.h"
#include "network.h"
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

AlwaysInline void w_out(std::string_view sv) {
    write(STDOUT_FILENO, sv.data(), sv.size());
}

AlwaysInline void w_err(std::string_view sv) {
    write(STDERR_FILENO, sv.data(), sv.size());
}

// so it seems like you have to call
// unref before you reuse AVPacket or AVFrame

void segv_handler(int /*unused*/) {
    w_err("Segmentation fault occurred. Please file a bug report on GitHub.\n");
    exit(EXIT_FAILURE); // NOLINT
}

// Idea:
// we could send the same chunk
// to different nodes if one is not doing it fast enough.
// To do this we could have some features in our vec of
//

template <class> inline constexpr bool always_false_v = false;

// all these call malloc:
// avcodec_send_frame():
// avbuffer_realloc
// av_packet_add_side_data
// avcodec_default_get_encoder_buffer()

// for chunked encoding

// ok so the next step would be to fix this null frame thing
// which probably has to do with incorrect usage of make_writable.
// also eventually we should somehow figure out how to add tests.

// also codec parameters don't really seem to be set correctly,
// not really sure why.

// we should probably also set timestamps properly.

// TODO ensure that two clients don't run in the same directory.
// For now at least tell the user that they shouldn't do this.

// allocates entire buffer upfront
// single threaded version
// TODO maybe implement as callback instead

// TODO see if it's possible to reuse an encoder
// by calling avcodec_open2 again.

// we need to make a version of this that doesn't just encode everything and
// flush the output at once

// TODO: possible optimization idea.
// For segments that need very high nskip value,
// fall back to sequential model and just give.
// that entire chunk to one client.

// runs single threaded mode
// TODO clean up string formatting, move it all to one centralized place
// format is
// "client_input_{idx}.mp4"
// TODO: use concat filter for this part of the code.

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

// TODO handle sigsegv as well
__attribute__((cold)) void sigint_handler(int /*unused*/) {
    exit(EXIT_FAILURE);
}

// TODO handle client disconnecting,
// add that back to the chunk queue

// for verify mode/dont trust we could request
// a hash of the segments to make sure.
// or perhaps just don't support untrusted workers.
// and just handle authentication to connect (password
// or whatever).

int try_parse_uint(unsigned int& result, const char* sp,
                   std::string_view argname) {
    std::string_view str = sp;
    auto [ptr, ec] =
        std::from_chars(str.data(), str.data() + str.size(), result, 10);

    if (ec == std::errc()) {
        if (ptr != (str.data() + str.size())) {
            (void)fprintf(stderr,
                          DIVIEN ": Error: Argument for " SVF ": argument "
                                 "contains non-digit characters\n",
                          SV(argname));
            return -1;
        }
        return 0;
    } else if (ec == std::errc::invalid_argument) {
        (void)fprintf(
            stderr, DIVIEN ": Error: Argument for " SVF ": invalid argument\n",
            SV(argname));
    } else if (ec == std::errc::result_out_of_range) {
        (void)fprintf(stderr,
                      DIVIEN ": Error: Argument for " SVF
                             ": value out of range\n",
                      SV(argname));
    }
    return -1;
}

void find_valid_opts(std::unordered_set<std::string_view>& params,
                     const AVClass* av_class) {
    DvAssert(av_class->option != nullptr);

    if (av_class->option) {
        params.reserve(64);
        // av_opt_show2(&av_class, nullptr, flags, 0);

        const AVOption* opt = nullptr;
        while ((opt = av_opt_next(&av_class, opt))) {
            // printf("Param: %s, %d\n", opt->name,
            //        opt->type == AV_OPT_TYPE_CONST);
            if (opt->type != AV_OPT_TYPE_CONST) {
                params.insert(std::string_view(opt->name));
            }
        }
    }

    // TODO we don't need this right?
    // void* iter = nullptr;
    // const AVClass* child;
    // while (child = av_opt_child_class_iterate(av_class, &iter)) {
    //     show_help_children(child, flags);
    // }
}

void print_help_encoder(const AVClass* av_class) {
    DvAssert(av_class->option != nullptr);

    if (av_class->option) {
        av_opt_show2(&av_class, nullptr, -1, 0);
    }
}

// TODO: thread priority + thread affinity CLI options
// TODO: allow more flexible -c:v placement. Just check
// if an option is a divien option instead and stop parsing
// encoder args if so.

// TODO: move argparse code to separate file

#define DIVIEN_VERSION "0.0.0 ALPHA (Unreleased)"

int main(int argc, const char* argv[]) {

    // TODO could implement checking on our side with allowed list of values

    if (argc < 2) {
        // TODO print this as help also
        // clang-format off
        w_err(DIVIEN ": must specify at least 2 args.\n"
              " Usage: ./DiViEn <mode> <args>\n"
              "        ./DiViEn [DiViEn args] -i <input> [-c:v <encoder>] [encoder args] <output>\n"
              "\n"
              "Available modes:\n"
              "    server <video>   Run centralized server for distributed encoding (WARNING: NOT FULLY FUNCTIONAL)\n"
              "    client           Run client that connects to DiViEn server instance (WARNING: NOT FULLY FUNCTIONAL)\n"
              "    (standalone)     Encode input video locally. This option is implied if no mode is specified.\n"
              "        args:\n"
              "          -i        <input_path>      Path to input video to encode [required]\n"
              "          -w        <num_workers>     Set number of workers (parallel encoder instances)\n"
              "          -threads  <num_threads>     Set number of threads per worker\n"
              "                                        NOTE: This does not always correlate to the encoder's threading\n"
              "                                        options. Prefer manually specifying encoder-specific options\n"
              "                                        if available.\n"
              "          -bsize    <num_frames>      Set frame buffer size (chunk size) for each worker\n"
              "          -c:v      <codec_name>      Set codec for encoding [default: libx264]\n"
              "          [encoder options]           List of arguments to pass to the encoder\n"
              );
        // clang-format on

        return -1;
    }
    DvAssert(argc >= 2);

#if defined(__unix__)
    struct sigaction sigIntHandler {};

    sigIntHandler.sa_handler = sigint_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, nullptr);
#endif

    // ok now we have our actual TCP server/client setup here.
    // next step AFTER this is to setup async
    // so that we can handle multiple clients at once easily.
    // Or at least multithreading or whatever.
    // Honestly yeah let's just use async.
    auto mode = std::string_view(argv[1]);
    try {
        if (mode == "--version" || mode == "-version") {
            w_out(DIVIEN " version " DIVIEN_VERSION "\n");
            return 0;
        }
        if (mode == "server") {
            if (argc < 3) {
                w_err(DIVIEN_ERR
                      "Must specify an input video for server mode.\n");
                return -1;
            }

            // DISTRIBUTED IDEA
            // use async file IO from ASIO, to avoid blocking other clients
            // from receiving data in buffers and stuff
            run_server_full(argv[2]);
        } else if (mode == "client") {
            // FrameAccurateWorkItem work{
            //     .low_idx = 26, .high_idx = 26, .nskip = 0, .ndecode = 50};
            // so it VERY SPECIFICALLY has to do with putting together frames
            // from multiple different sources.
            // Perhaps we can use the concat FILTER to deal with this problem.
            // I hope that would work.
            // encode_frame_range(work, "random.mp4");

            run_client_full();
        } else {
            // interpret as standalone mode otherwise
            // TODO perhaps ensure duplicate arguments don't exist

            // optional params
            const char* output_path_s = nullptr;
            const char* input_path_s = nullptr;
            const char* encoder_name_s = nullptr;

            const char* num_workers_s = nullptr;
            const char* threads_per_worker_s = nullptr;
            const char* framebuf_size_s = nullptr;
            const char** ff_enc_first_param = nullptr;
            size_t ff_enc_n_param_pairs = 0;

            // so we're going to parse the input file first, then output file
            // options, then output file name

            // parse the rest of the options here

            // TODO swap out unordered_set impl to something lighter weight
            std::unordered_set<std::string_view> valid_opts{};
            const AVCodec* codec = nullptr;

            for (size_t arg_i = 1; arg_i < (size_t)argc; arg_i++) {

                // this is ugly but at least it works

#define LENGTH_CHECK(store_variable)                                           \
    {                                                                          \
        if (arg_i + 1 >= (size_t)argc) [[unlikely]] {                          \
            goto print_err;                                                    \
        }                                                                      \
        (store_variable) = argv[++arg_i];                                      \
    }

                auto arg_sv = std::string_view(argv[arg_i]);
                // TODO: deduplicate code somehow?
                // TODO: perhaps we should detect incorrect command line/missing
                // values by checking the next argument is another command.
                // TODO: perhaps a deduplication technique would be to store
                // the variables in an array of strings as well, so all
                // information about the option comes from the same index.
                if (arg_sv == "-i") {
                    LENGTH_CHECK(input_path_s);
                } else if (arg_sv == "-w") {
                    LENGTH_CHECK(num_workers_s);
                } else if (arg_sv == "-threads") {
                    LENGTH_CHECK(threads_per_worker_s);
                } else if (arg_sv == "-bsize") {
                    LENGTH_CHECK(framebuf_size_s);
                } else if (arg_sv == "-c:v" || arg_sv == "-vcodec" ||
                           arg_sv == "-codec:v") {
                    if (encoder_name_s != nullptr) [[unlikely]] {
                        printf(DIVIEN_ERR "Encoder already specified, cannot "
                                          "provide " SVF "\n",
                               SV(arg_sv));
                        return -1;
                    }

                    LENGTH_CHECK(encoder_name_s);

                    codec = avcodec_find_encoder_by_name(encoder_name_s);
                    if (!codec) [[unlikely]] {
                        printf(DIVIEN_ERR "No codec '%s' found.\n",
                               encoder_name_s);
                        return -1;
                    }
                    find_valid_opts(valid_opts, codec->priv_class);
                    DbgDvAssert(valid_opts.size() != 0);
                } else {
                    // arg_i is currently set to some other option
                    // check if it's an encoder option
                    // no more opts
                    // last argument, interpret as output filename and stop
                    // parsing
                    if (arg_i == (size_t)argc - 1) {
                        output_path_s = argv[arg_i];
                        break;
                    }

                    // do not allow additional arguments if there was no
                    // explicit encoder specified

                    // TODO check for this better
                    if (encoder_name_s == nullptr) {
                        DIVIEN_ABORT("Arguments for encoder were provided "
                                     "but no encoder was specified with -c:v");
                    }

                    DbgDvAssert(valid_opts.size() != 0);

                    // guaranteed that arg_i is not the last argument here
                    // since we handled that before

                    // more options were specified
                    ff_enc_n_param_pairs = ((size_t)argc - 2) - arg_i + 1;
                    ff_enc_first_param = &argv[arg_i];
                    if (ff_enc_n_param_pairs & 1) {
                        DIVIEN_ABORT(
                            DIVIEN_ERR
                            "Mismatched parameters: odd number of parameters "
                            "specified for encoder.\n"
                            "  Encoder arguments must be a list of key-value "
                            "pairs.\n  All parameters after DiViEn options and "
                            "before the output file are interpreted as encoder "
                            "arguments.");
                    }

                    for (size_t offi = 0; offi < ff_enc_n_param_pairs;
                         offi += 2) {
                        DbgDvAssert(offi + 1 < ff_enc_n_param_pairs);
                        const char* key = ff_enc_first_param[offi];
                        // const char* value = ff_enc_first_param[offi + 1];
                        if (key[0] != '-') {
                            DIVIEN_ABORT("Keys for encoder options must start "
                                         "with '-'.");
                        }
                        DbgDvAssert(codec != nullptr);
                        if (!valid_opts.contains(std::string_view(key + 1))) {
                            printf(DIVIEN_ERR "Invalid option '%s' for encoder "
                                              "%s.\n\nValid parameters:\n",
                                   key + 1, encoder_name_s);
                            print_help_encoder(codec->priv_class);
                            return -1;
                        }
                    }
                    // TODO clean this up
                    ff_enc_n_param_pairs /= 2;

                    output_path_s = argv[argc - 1];
                    break;
                }
                continue;
            print_err:
                printf(DIVIEN_ERR "No argument specified for " SVF ".\n",
                       SV(arg_sv));
                return -1;
            }

            if (encoder_name_s == nullptr) {
                encoder_name_s = "libx264";
            }
            // TODO validate that file ends with an extension
            if (output_path_s == nullptr) {
                w_err(DIVIEN_ERR "No output path specified. Please specify the "
                                 "output path after the encoder arguments.\n");
                return -1;
            }
            if (input_path_s == nullptr) {
                w_err(DIVIEN_ERR "No input path provided. Please specify the "
                                 "input path with -i.\n");
                return -1;
            }

            printf("Using output file '%s'\n", output_path_s);

#define PARSE_OPTIONAL_ARG(data_var, string_var, arg_name)                     \
    {                                                                          \
        if ((string_var) != nullptr) {                                         \
            if (try_parse_uint(data_var, string_var, arg_name) < 0)            \
                [[unlikely]] {                                                 \
                return -1;                                                     \
            }                                                                  \
        }                                                                      \
    }

#define VALIDATE_ARG_NONZERO(data_var, arg_name)                               \
    if ((data_var) == 0) [[unlikely]] {                                        \
        w_err(DIVIEN ": Error: " arg_name " cannot be 0\n");                   \
        return -1;                                                             \
    }

            // default values
            unsigned int num_workers = 4;
            unsigned int threads_per_worker = 4;
            unsigned int chunk_size = 250;

            // Does assigning string literal to variable give that
            // proper (static) lifetime? Need to know for macro purpose.
            // Ideally we assign to string_view though.
            PARSE_OPTIONAL_ARG(num_workers, num_workers_s, "-w");
            VALIDATE_ARG_NONZERO(num_workers, "-w");
            // perhaps just rename this option -threads?
            PARSE_OPTIONAL_ARG(threads_per_worker, threads_per_worker_s,
                               "-threads");
            VALIDATE_ARG_NONZERO(threads_per_worker, "-threads");
            PARSE_OPTIONAL_ARG(chunk_size, framebuf_size_s, "-bsize");
            VALIDATE_ARG_NONZERO(chunk_size, "-bsize");

            // Now we need to validate the input we received.
            // TODO: move everything for CLI parsing into its own function.

            // TODO: we need to assert what was null and what wasn't.

            DvAssert(input_path_s != nullptr);
            DvAssert(output_path_s != nullptr);

            // TODO rename variables to be less confusing
            printf("Using %u workers (%u threads per worker), chunk size "
                   "= %u "
                   "frames per worker\nRunning in standalone mode [%s]\n",
                   num_workers, threads_per_worker, chunk_size, encoder_name_s);

            // TODO make sure this doesn't throw exceptions.
            auto vdec =
                DecodeContext::open(input_path_s, chunk_size * num_workers);
            if (std::holds_alternative<DecoderCreationError>(vdec)) {
                auto err = std::get<DecoderCreationError>(vdec);
                // TODO: make this function format into buffer or something, for
                // more specific error messages on AVError. Need to deduplicate
                // a bunch of code.
                auto m = err.errmsg();
                printf("Failed to open input file '%s': " SVF "\n",
                       input_path_s, SV(m));
                return -1;
            }

            auto& dc = std::get<DecodeContext>(vdec);

            DvAssert(dc.demuxer != nullptr);
            av_dump_format(dc.demuxer, dc.video_index, input_path_s, 0);

            av_log_set_level(AV_LOG_WARNING);

            EncoderOpts e_opts(encoder_name_s, ff_enc_first_param,
                               ff_enc_n_param_pairs);

            // TODO make params and param pairs all a part of encoder options
            // struct that's passed around
            DvAssert(chunked_encode_loop(e_opts, input_path_s, output_path_s,
                                         dc, num_workers, chunk_size,
                                         threads_per_worker) == 0);

            (void)fflush(stderr);
            (void)fflush(stdout);
        }

    } catch (std::exception& e) {
        printf(DIVIEN_ERR "Exception occurred: %s\n", e.what());
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
