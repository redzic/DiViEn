#pragma once

// TODO maybe change this to "DiViEn: "?
#define DIVIEN "DiViEn"
#define DIVIEN_ERR "DiViEn: Error: "
#define DIVIEN_ABORT(msg)                                                      \
    {                                                                          \
        w_err(DIVIEN_ERR msg "\n");                                            \
        return -1;                                                             \
    }
