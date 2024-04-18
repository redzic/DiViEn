#pragma once

#define AlwaysInline __attribute__((always_inline)) inline

#define SV(sv_var) (int)(sv_var).size(), (sv_var).data()

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
