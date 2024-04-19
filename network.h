#pragma once

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

#include <vector>

#include "encode.h"
#include "segment.h"
#include "util.h"

// 128 chars is a good size I think
// Could even be reduced to 64
using fmt_buf = std::array<char, 128>;

using asio::awaitable;
using asio::co_spawn;
using asio::detached;
using asio::use_awaitable;
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

// TODO move networking code to another flie.

// TODO: file not existing errors need to be properly displayed to user,
// not a runtime segfault and only debug assert.

[[nodiscard]] awaitable<size_t> socket_send_file(tcp_socket& socket,
                                                 const char* filename,
                                                 asio::error_code& error);

// TODO come up with a better name for this function and the write version
// This is just an ugly name man.
// Returns number of bytes received
[[nodiscard]] awaitable<size_t> socket_recv_file(tcp_socket& socket,
                                                 const char* dumpfile,
                                                 asio::error_code& error);

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

// TODO do not use any_io_executor as it's the defualt polymorphic
// type
// try to use io_context::executor_type instead.

// memory leaks happen when I do ctrl+C...
// Is that supposed to happen?

// also TODO need to add mechanism to shutdown server

// there's an issue here man.

// alright well this does seem to work
// and the io_context
// is there a better approach though?
// kills the io_context when all work is finished.
// https://en.cppreference.com/w/cpp/thread/condition_variable/wait

// god this code is messy
// TODO return nb_segments as value (and all similar functions in code)
// sadly there's some kind of bug with segmenting that huge video.
// TODO it might be because we're not using av_make_frame_writable.
// Check if decoder is failing or just encoder.

FinalWorklist server_prepare_work(const char* source_file,
                                  unsigned int& nb_segments);
// TODO remove unused arguments
// TODO move all networking code to its own file

awaitable<void> run_server(asio::io_context& context, const char* source_file,
                           ServerData& state);

// So the client doesn't even have to be async.
// that can just be regular old sync.

template <typename F> struct OnReturn {
    F f;

    explicit OnReturn(F f_) : f(f_) {}
    DELETE_DEFAULT_CTORS(OnReturn)
    ~OnReturn() { f(); }
};

// TODO. Perhaps the client should run, because it might lower throughput.

awaitable<void> run_client(asio::io_context& io_context, tcp_socket socket,
                           asio::error_code& error);
void run_server_full(const char* from_file);

void run_client_full();