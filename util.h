#pragma once

#define DELETE_DEFAULT_CTORS(MacroArgStructName)                               \
    MacroArgStructName() = delete;                                             \
    MacroArgStructName(MacroArgStructName&&) = delete;                         \
    MacroArgStructName(MacroArgStructName&) = delete;                          \
    MacroArgStructName& operator=(const MacroArgStructName&) = delete;         \
    MacroArgStructName& operator=(const MacroArgStructName&&) = delete;

#define DELETE_COPYMOVE_CTORS(MacroArgStructName)                              \
    MacroArgStructName(MacroArgStructName&&) = delete;                         \
    MacroArgStructName(MacroArgStructName&) = delete;                          \
    MacroArgStructName& operator=(const MacroArgStructName&) = delete;         \
    MacroArgStructName& operator=(const MacroArgStructName&&) = delete;
