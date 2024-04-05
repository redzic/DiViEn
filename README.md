# DiViEn

Local and distributed chunked video encoding.

This project is still under development. There are
many known issues (mainly with how timestamps are set).

Distributed encoding does not work properly yet,
however it is under development.

Only supports the latest build of FFmpeg libaries (as of now, 6.1).
This is due to hacks/workarounds used when dealing with
libav. Other versions of FFmpeg may work, but they are untested.

If you would like to discuss the project, report issues, or
give suggestions, you can join the discord server: https://discord.gg/NEwN2sTDbj

## Compilation

This project depends on ASIO (non Boost version), FFmpeg 6.1, and optionally io-uring on linux (disabled by default). It requires a C++20 compiler.

To compile this project, run:

```
git clone https://github.com/redzic/DiViEn && cd DiViEn
mkdir build && cd build
cmake ..
cmake --build .
```

## Usage

To run the local chunked encoder:

```
./DiViEn standalone <path/to/input.mp4>
```

The output will be in `standalone_output.mp4`.

A proper CLI will be implemented later.

Note that this might be slow on your machine because of the default amount of workers. (See: Current Limitations.)
For now, please edit the variables in the source code and recompile to use different amounts of workers and threads (See the file decode.h, variables NUM_WORKERS, THREADS_PER_WORKER, and CHUNK_FRAME_SIZE). Like I said I will get to implementing a proper CLI later.

The same applies for encoding parameters.

The output file may not be fully compatible with some video players at the moment due to
a lack of proper timestamps being set. This should be fixed in the future.

Distributed encoding currently is implemented in a basic form, but contains bugs. These should be fixed in the future.

### Current Limitations 

- Many things are hard coded into the source code,
like the number of workers, frame buffer size, and encoding
parameters. This will be changed as a proper CLI is implemented.

- Currently, only fixed sized chunks are supported. Scene detection is not currently implemented.
