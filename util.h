#pragma once

#define AlwaysInline __attribute__((always_inline)) inline

#define SV(sv_var) (int)(sv_var).size(), (sv_var).data()
#define SVF "%.*s"

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

#ifdef NDEBUG
constexpr inline void DvAssert(bool /*unused*/) {}
#define DvAssert2 DvAssert
#define DbgDvAssert(expr) ((void)0)
#else // DEBUG
#include <cassert>
#include <libassert/assert.hpp>
// Regular assertion (side effects exist in debug mode), most common use case
#define DvAssert DEBUG_ASSERT
// Workaround for certain unsupported cases in libassert (e.g., bit fields)
#define DvAssert2 assert
// Debug assertion, expression and side effects deleted entirely in Release
// mode.
#define DbgDvAssert DEBUG_ASSERT
#endif
