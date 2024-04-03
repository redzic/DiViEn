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

This project depends on ASIO (non Boost version), FFmpeg 6.1, and io-uring (this
limitation will be lifted soon). It requires a C++20 compiler.

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

### Current Limitations 

- Many things are hard coded into the source code,
like the number of workers, frame buffer size, and encoding
parameters. This will be changed as a proper CLI is implemented.

- Currently, only fixed sized chunks are supported. Scene detection is not currently implemented.
