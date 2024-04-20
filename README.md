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
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

## Usage

To run the local chunked encoder (standalone mode):

Basic usage:

```
./DiViEn -i <path/to/input.mp4> <output_file>
```

Specify arguments (e.g., 8 workers, 120 frame buffer size per worker, libaom-av1, custom encoding params):

```
./DiViEn -i <path/to/input.mp4> -w 8 -bsize 120 -c:v libaom-av1 -crf 30 -cpu-used 6 <output_file>
```

Specify custom arguments and threading options with x265:

```
./DiViEn -i <path/to/input.mp4> -w 8 -threads 4 -c:v libx265 -crf 25 -preset veryfast -x265-params pools=2 <output_file>
```

All parameters after `-c:v <encoder>` are interpreted as arguments to the encoder, except for the last parameter which is interpreted as the output file.

Note that the `-threads` (threads per worker) option does not always correlate with encoder-specific options. It functions the same as the similarly-named `-threads` option in ffmpeg. Prefer encoder-specific options for threading if available. Note that decoder threading is always set to automatic selection, regardless of any command-line options specified.

The output file may not be fully compatible with some video players at the moment due to
a lack of proper timestamps being set. This should be fixed in the future.

Distributed encoding currently is implemented in a basic form. It is in early stages, so if you find any issues with it please report them.

### Current Limitations

- Currently, only fixed sized chunks are supported. Scene detection is not currently implemented.
