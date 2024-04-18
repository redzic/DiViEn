#pragma once

#include "util.h"
#include <array>

extern "C" {
#include <libavutil/error.h>
}

AlwaysInline auto av_strerr(int averror) {
    std::array<char, AV_ERROR_MAX_STRING_SIZE> errbuf;
    av_make_error_string(errbuf.data(), errbuf.size(), averror);
    return errbuf;
}
