#include "network.h"

#include "resource.h"
#include "timing.h"

#include <array>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <pthread.h>
#include <unistd.h>
#include <unordered_set>

constexpr size_t TCP_BUFFER_SIZE = 2048z * 32 * 4; // 256 Kb buffer

// Returns number of bytes read.
// TODO make async.
[[nodiscard]] awaitable<size_t> socket_send_file(tcp_socket& socket,
                                                 const char* filename,
                                                 asio::error_code& error) {
    printf("Called socket_send_file\n");

    // Open a file in read mode
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

[[nodiscard]] awaitable<void>
handle_conn(tcp_socket socket, unsigned int conn_num, ServerData& state) {
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

                // work = state.work.client_chunks.at(15);
                // get some work to be done
                // TODO: write some code to double check frame ranges and
                // compute frame hashes or whatever.
                // but we will need to validate it with some external
                // thing.
                // I think it might be something in our decode loop logic.

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

awaitable<void> kill_context(asio::io_context& context) {
    context.stop();

    co_return;
}

static void server_stopper_thread(asio::io_context& context, ServerData& data) {
    std::unique_lock<std::mutex> lk(data.tk_cv_m);
    data.tk_cv.wait(lk,
                    [&]() { return data.orig_work_size == data.chunks_done; });

    co_spawn(context, kill_context(context), detached);
}

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

awaitable<void> run_server(asio::io_context& context, const char* source_file,
                           ServerData& state) {

    asio::error_code error;

    tcp_acceptor acceptor(context, {tcp::v4(), 7878});

    printf("[Async] Listening for connections...\n");
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
        co_spawn(context, handle_conn(std::move(socket), conn_num, state),
                 detached);

        printf("[TCP] Connection %d closed\n", conn_num);
    }

    printf("Returning from run_server()\n");
    co_return;
}

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

            auto already_have =
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
        (void)snprintf(output_buf.data(), output_buf.size(),
                       "enc_%u_%u_%u_%u.mp4", work.low_idx, work.high_idx,
                       work.nskip, work.ndecode);

        encode_frame_range(work, output_buf.data());

        co_await DISPLAY_SPEED(
            socket_send_file(socket, output_buf.data(), error));

        printf("Uploaded %s to server\n", output_buf.data());
    }
    co_return;
}

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
                    .work_list_mutex = {},
                    .tk_cv = {},
                    .tk_cv_m = {}};

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

void run_client_full() {
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
        printf("Error occurred connecting to server: %s\n", msg.c_str());
        exit(EXIT_FAILURE);
    }

    co_spawn(io_context, run_client(io_context, std::move(socket), error),
             detached);

    io_context.run();
}