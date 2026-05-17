#pragma once
#ifdef USE_METAL

#include <cstddef>
#include <cstdint>

// Opaque context — implementation is Objective-C++ (metal_rs.mm).
struct MetalRS;

// Allocate a Metal RS context for the given recovery block count and chunk size.
// Returns a context even if no GPU is available; check metal_rs_available().
MetalRS* metal_rs_create(uint32_t recovery_block_count, size_t chunksize);
void     metal_rs_destroy(MetalRS* ctx);
bool     metal_rs_available(MetalRS* ctx);

// Zero the GPU output buffer. Call once at the start of each ProcessData pass.
void metal_rs_clear_output(MetalRS* ctx);

// Accumulate one source block into all recovery blocks on the GPU.
// src       : inputbuffer, blocklength bytes
// factors   : leftmatrix column for this source block, rec_count uint16_t values
// Synchronous: returns after GPU has finished.
void metal_rs_process_block(MetalRS* ctx,
                             const void*     src,
                             size_t          blocklength,
                             const uint16_t* factors,
                             uint32_t        rec_count);

// Copy GPU output buffer to cpu_dst (chunksize * rec_count bytes,
// recovery block n at offset n*chunksize — same layout as par2creator's outputbuffer).
void metal_rs_get_output(MetalRS* ctx, void* cpu_dst);

#endif // USE_METAL
