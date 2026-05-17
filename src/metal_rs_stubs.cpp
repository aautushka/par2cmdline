#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

// Weak stub implementations of the metal_rs API.
// These are compiled into libpar2.a so that any binary linking libpar2.a
// can resolve metal_rs_* symbols without needing the Metal framework.
//
// When a binary also links metal_rs.mm (e.g. the par2 binary and metal_rs_test),
// the strong definitions in metal_rs.mm override these weak stubs.
// When only libpar2.a is linked (e.g. unit test programs), these stubs
// satisfy the linker and metal_rs_available() returns false, so
// par2creator.cpp falls back to the CPU path automatically.

#ifdef USE_METAL

#include "metal_rs.h"
#include <cstddef>
#include <cstdint>

__attribute__((weak)) MetalRS* metal_rs_create(uint32_t, size_t)           { return nullptr; }
__attribute__((weak)) void     metal_rs_destroy(MetalRS*)                   {}
__attribute__((weak)) bool     metal_rs_available(MetalRS*)                 { return false; }
__attribute__((weak)) void     metal_rs_clear_output(MetalRS*)              {}
__attribute__((weak)) void     metal_rs_process_block(MetalRS*, const void*,
                                   size_t, const uint16_t*, uint32_t)       {}
__attribute__((weak)) void     metal_rs_get_output(MetalRS*, void*)         {}

#endif // USE_METAL
