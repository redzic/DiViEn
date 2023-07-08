#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <optional>
#include <unistd.h>
#include <variant>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
}

inline void w_stdout(std::string_view sv) {
    write(STDOUT_FILENO, sv.data(), sv.size());
}
inline void w_stderr(std::string_view sv) {
    write(STDERR_FILENO, sv.data(), sv.size());
}

template <typename T, auto Alloc, auto Free> auto make_managed() {
    return std::unique_ptr<T, decltype([](T* ptr) { Free(&ptr); })>(Alloc());
}

struct DecoderCreationError {
    enum DCErrorType : uint8_t {
        AllocationFailure,
        NoVideoStream,
        NoDecoderAvailable,
        AVError
    } type;
    int averror = 0;

    [[nodiscard]] constexpr std::string_view errmsg() const {
        static constexpr std::string_view errmsg_sv[] = {
            [AllocationFailure] = "Allocation Failure in decoder construction",
            [NoVideoStream] = "No video stream exists in input file",
            [NoDecoderAvailable] = "No decoder available for codec",
            [AVError] = "Unspecified AVError occurred",
        };

        // if (this->type != AVError) {
        return errmsg_sv[this->type];
        // }

        // return "";
    }
};

// VideoDecodeContext
struct VidDecCtx {
    // all fields are owned and must be non-null
    AVFormatContext* demuxer;
    AVStream* stream;
    AVCodecContext* decoder;

    AVPacket* pkt;
    AVFrame* frame;

    // it seems that using std::variant cuases a bun
    [[nodiscard]] static std::variant<VidDecCtx, DecoderCreationError>
    open(const char* url) {
        auto pkt = make_managed<AVPacket, av_packet_alloc, av_packet_free>();
        auto frame = make_managed<AVFrame, av_frame_alloc, av_frame_free>();

        // this should work with the way smart pointers work right?
        if ((pkt == nullptr) || (frame == nullptr)) {
            return {DecoderCreationError{
                .type = DecoderCreationError::AllocationFailure}};
        }

        AVFormatContext* raw_demuxer = nullptr;

        // avformat_open_input automatically frees on failure so we construct
        // the smart pointer AFTER this expression.
        {
            int ret = avformat_open_input(&raw_demuxer, url, nullptr, nullptr);
            if (ret < 0) {
                return {DecoderCreationError{
                    .type = DecoderCreationError::AVError, .averror = ret}};
            }
        }

        assert(raw_demuxer != nullptr);
        auto demuxer =
            std::unique_ptr<AVFormatContext, decltype([](AVFormatContext* ctx) {
                                avformat_close_input(&ctx);
                            })>(raw_demuxer);

        avformat_find_stream_info(demuxer.get(), nullptr);

        // find stream idx of video stream
        int stream_idx = [](AVFormatContext* demuxer) {
            for (unsigned int stream_idx = 0; stream_idx < demuxer->nb_streams;
                 stream_idx++) {
                if (demuxer->streams[stream_idx]->codecpar->codec_type ==
                    AVMEDIA_TYPE_VIDEO) {
                    return static_cast<int>(stream_idx);
                }
            }
            return -1;
        }(demuxer.get());

        if (stream_idx < 0) {
            return {DecoderCreationError{
                .type = DecoderCreationError::NoVideoStream,
            }};
        }

        // index is stored in AVStream->index
        auto* stream = demuxer->streams[stream_idx];

        const auto* codec = avcodec_find_decoder(stream->codecpar->codec_id);
        if (codec == nullptr) {
            return {DecoderCreationError{
                .type = DecoderCreationError::NoDecoderAvailable,
            }};
        }

        auto decoder =
            std::unique_ptr<AVCodecContext, decltype([](AVCodecContext* ctx) {
                                avcodec_free_context(&ctx);
                            })>(avcodec_alloc_context3(codec));

        {
            int ret =
                avcodec_parameters_to_context(decoder.get(), stream->codecpar);
            if (ret < 0) {
                return {DecoderCreationError{
                    .type = DecoderCreationError::AVError, .averror = ret}};
            }
        }

        // set automatic threading
        decoder->thread_count = 0;

        return VidDecCtx(demuxer.release(), stream, decoder.release(),
                         pkt.release(), frame.release());
    }

    // For some seemingly inexplicable reason, having this destructor present
    // seems to cause some extremely weird issues to happen where a nonzero
    // exit code is returned from main. This doesn't happen when deleting the
    // destructor.
    ~VidDecCtx() {
        av_frame_free(&frame);
        av_packet_free(&pkt);
        avcodec_free_context(&decoder);
        avformat_close_input(&demuxer);
    }

    VidDecCtx(auto demuxer_, auto stream_, auto decoder_, auto pkt_,
              auto frame_)
        : demuxer(demuxer_), stream(stream_), decoder(decoder_), pkt(pkt_),
          frame(frame_) {}
};

// so it seems like you have to call
// unref before you reuse AVPacket or AVFrame

void segvHandler(int) {
    w_stderr("Segmentation Fault\n");
    std::quick_exit(EXIT_FAILURE);
}

int main(int argc, char** argv) {
    signal(SIGSEGV, segvHandler);

    if (argc != 2) {
        w_stderr("scenedetect-cpp: invalid number of arguments\n"
                 "   usage: scenedetect-cpp  <video_file>\n");
        return -1;
    }

    // can assume argv[1] is now available
    char* url = argv[1];

    // bro how on earth is the exit code being set to
    // something other than 0???
    auto vdec = VidDecCtx::open(url);

    // so as soon as you use something like std::cout, the binary size increases
    // greatly...

    // so we should probably find a way to not use things that increase the
    // binary size a lot...

    if (std::holds_alternative<VidDecCtx>(vdec)) {
        w_stdout("Decoder object constructed\n");
    } else if (std::holds_alternative<DecoderCreationError>(vdec)) {
        auto error = std::get<DecoderCreationError>(vdec);

        if (error.type == DecoderCreationError::AVError) {
            std::array<char, 512> errbuf{};
            av_make_error_string(errbuf.data(), errbuf.size(), error.averror);
            std::cerr << "Decoder failed to construct: " << errbuf.data()
                      << '\n';
        } else {
            // this use of cerr causes a shit ton of extra code to be generated.
            // so remove later if possible.
            std::cerr << "Decoder failed to construct: " << error.errmsg()
                      << '\n';
        }

    } else {
        // TODO perhaps try to figure out a way to do this that doesn't require
        // this hack. Without this the compiler generates unnecessary code in
        // the variant type check and doesn't optimize things perfectly.
        __builtin_unreachable();
    }
}
