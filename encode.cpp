#include "encode.h"

#include "decode.h"
#include "ffutil.h"
#include "resource.h"

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
        int ret = av_opt_set(avcc->priv_data, key + 1, value, 0);
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
            fprintf(stderr, "\n\nWARNING: Failed to set %s=%s: %s\n\n", key,
                    value, err);
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
    DvAssert(enc_ctx != nullptr);
    int ret = avcodec_send_frame(enc_ctx, frame);

    // I think it has to do with some of the metadata not being the same...
    // compared to the previous frames. Does that have to do with segmenting?
    // if the issue doesn't happen on chunks that are fixed up, then
    // perhaps there's a pattern there...
    // But then why does it happen only on like the 3RD frame and only on
    // SOME encoders?
    if (frame != nullptr) {
        DvAssert(avframe_has_buffer(frame));
    }

    if (ret < 0) {
        // TODO deduplicate, make macro or function to do this
        if (frame == nullptr) {
            printf("error sending flush frame to encoder: ");
        } else {
            printf("error sending frame to encoder: ");
        }
        printf("encoder error: %s\n", av_strerr(ret).data());

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

        DvAssert(fwrite(pkt->data, 1, pkt->size, ostream) == (size_t)pkt->size);

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

    for (auto* frame : framebuf) {
        // required
        frame->pict_type = AV_PICTURE_TYPE_NONE;
        encode_frame(encoder.avcc, frame, encoder.pkt, file.get(),
                     state.nb_frames_done);
    }
    // need to send flush packet after we're done
    encode_frame(encoder.avcc, nullptr, encoder.pkt, file.get(),
                 state.nb_frames_done);

    return 0;
}

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

    for (uint32_t idx = data.low_idx; idx <= data.high_idx; idx++) {

        std::array<char, 128> input_fname;
        (void)snprintf(input_fname.data(), input_fname.size(),
                       "client_input_%d.mp4", idx);

        printf("=========== NEW SUBCHUNK. OPENING %s FOR READING.\n",
               input_fname.data());

        auto dres = DecodeContext::open(input_fname.data(), 0);

        // perhaps we could initialize all these decoders at the same time...
        // to save time.
        // TODO reuse underlying frame buffer

        std::visit(
            [&, ofptr = ofptr.get()](auto&& dc) {
                using T = std::decay_t<decltype(dc)>;

                if constexpr (std::is_same_v<T, DecodeContext>) {
                    // TODO split loop if possible
                    if (idx == data.low_idx) {
                        // means we first need to decode nskip frames
                        // initially, frames are not writable. Because they
                        // have
                        // not been properly allocated yet, their actual
                        // buffers
                        // I mean. The decoder allocates those buffers.
                        for (uint32_t nf = 0; nf < data.nskip; nf++) {
                            AVFrame* frame = av_frame_alloc();

                            // first iteration of loop
                            DvAssert(decode_next(dc, frame) == 0);
                            if (nf == 0) {
                                // TODO use proper amount of threads
                                encoder.initialize_codec(frame, 1,
                                                         DEFAULT_ENCODER);
                                enc_was_init = true;
                            }
                            av_frame_unref(frame);
                        }
                    } else {
                        DvAssert(enc_was_init);
                    }

                    // TODO allow configurable frame size when constructing
                    // decoder, to avoid wasting memory
                    while (nleft > 0) {
                        printf("%d frames left\n\n", nleft);

                        AVFrame* frame = av_frame_alloc();

                        int ret = 0;
                        if ((ret = decode_next(dc, frame)) != 0) {
                            printf("Decoder unexpectedly returned error: %s\n",
                                   av_strerr(ret).data());
                            break;
                        }

                        if (!enc_was_init) [[unlikely]] {
                            // TODO use proper amoutn of threads
                            encoder.initialize_codec(frame, 1, DEFAULT_ENCODER);
                            enc_was_init = true;
                        }

                        DvAssert(frame != nullptr);
                        DvAssert(avframe_has_buffer(frame));
                        // so the issue is not that the frame doesn't have a
                        // buffer, hmm...
                        // DvAssert(av_frame_make_writable(frame) == 0);
                        printf("Sending frame... Is writable? %d\n",
                               av_frame_is_writable(frame));
                        DvAssert(encode_frame(encoder.avcc, frame, encoder.pkt,
                                              ofptr, nframes_done) == 0);
                        printf("frame= %u\n", nframes_done.load());

                        av_frame_unref(frame);

                        nleft--;
                    }
                } else {
                    printf("Decoder failed to open for input file '%s'\n",
                           input_fname.data());
                }
            },
            dres);
    }

    DvAssert(ofptr.get() != nullptr);
    printf("Sending flush packet...\n");
    // why does this return an error?
    encode_frame(encoder.avcc, nullptr, encoder.pkt, ofptr.get(), nframes_done);
}
