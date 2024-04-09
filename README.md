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

To specify custom encoder arguments, use the `-ff` option, followed by a list of ffmpeg-style arguments, followed by the delimiter `--`. After the delimiter, arguments will be interpreted as arguments to DiViEn. The delimiter can be omitted if `-ff` was the last option specified.

Specify arguments (e.g., 8 workers, 120 frame buffer size per worker, libaom-av1, custom encoding params):

```
./DiViEn -i <path/to/input.mp4> -w 8 -c:v libaom-av1 -ff -crf 30 -cpu-used 6 -- -bsize 120
```

Specify custom arguments and threading options with x265:

```
./DiViEn -i <path/to/input.mp4> -w 8 -c:v libx265 -ff -crf 25 -preset veryfast -x265-params pools=none:frame-threads=2
```

Note that the `-tpw` (threads per worker) option does not always correlate with encoder-specific options. `-tpw` corresponds with the `-threads` option in ffmpeg (when applied to the encoder). Prefer encoder-specific options for threading if available. Note that decoder threading is always set to automatic selection, regardless of any command-line options specified.

The template for the output file name is `<input_file_name>_<encoder_name>.mp4`. DiViEn will automatically create a temporary folder for the chunks. The temporary folder is not deleted after encoding, and if the folder already exists when DiViEn runs, DiViEn will overwrite the chunks. (This may change in the future.)

The output file may not be fully compatible with some video players at the moment due to
a lack of proper timestamps being set. This should be fixed in the future.

Distributed encoding currently is implemented in a basic form, but contains bugs. These should be fixed in the future.

### Current Limitations

- Currently, only fixed sized chunks are supported. Scene detection is not currently implemented.
