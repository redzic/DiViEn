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

To run the local chunked encoder (standalone mode):

Basic usage:

```
./DiViEn -i <path/to/input.mp4>
```

Specify arguments (e.g., 8 workers, 4 threads per worker, libaom-av1, custom encoding params):

```
./DiViEn -i <path/to/input.mp4> -w 8 -c:v libaom-av1 -ff -crf 30 -cpu-used 6 -- -tpw 4
```

Specify custom arguments and threading options with x265 (`--` can be omitted if `-ff` is at the end of the argument list):

```
./DiViEn -i <path/to/input.mp4> -w 8 -c:v libx265 -ff -crf 25 -preset veryfast -x265-params pools=none:frame-threads=2
```

Note that the `-tpw` option does not always correlate with encoder-specific options. `-tpw` corresponds with the `-threads` option in ffmpeg. Prefer encoder-specific options for threading if available.

The output will be in `standalone_output.mp4`.

The output file may not be fully compatible with some video players at the moment due to
a lack of proper timestamps being set. This should be fixed in the future.

Distributed encoding currently is implemented in a basic form, but contains bugs. These should be fixed in the future.

### Current Limitations

- Currently, only fixed sized chunks are supported. Scene detection is not currently implemented.
