#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#ifdef USE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_rs.h"
#include <cstring>

// ---------------------------------------------------------------------------
// GF(2^16) tables — same generator as galois.h (0x1100B), rebuilt here to
// avoid touching that file's protected static member.
// ---------------------------------------------------------------------------
static void build_gf16_tables(uint16_t log_tab[65536], uint16_t alog_tab[65536])
{
    const uint32_t Limit     = 65535;
    const uint32_t Generator = 0x1100B;
    uint32_t b = 1;
    for (uint32_t l = 0; l < Limit; l++) {
        log_tab[b]  = (uint16_t)l;
        alog_tab[l] = (uint16_t)b;
        b <<= 1;
        if (b & 65536) b ^= Generator;
    }
    log_tab[0]      = (uint16_t)Limit;
    alog_tab[Limit] = 0;
}

// ---------------------------------------------------------------------------
// Tiled Metal shader — compiled at runtime to avoid a separate .metallib step.
//
// Key design: each thread owns one (word_idx, rec_idx) pair and loops over
// all src_in_tile source blocks, accumulating GF(2^16) products in a register.
// The output buffer (buf_dst) is touched exactly once per dispatch — one read
// and one XOR-write — regardless of how many source blocks are in the tile.
// This is the core of the memory-traffic reduction vs the old per-block kernel.
// ---------------------------------------------------------------------------
static const char* kShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void rs_process_tiled(
    device const uint16_t* src_tile     [[buffer(0)]],  // src_in_tile * stride_words
    device       uint16_t* dst          [[buffer(1)]],  // rec_count * stride_words
    device const uint16_t* factors_tile [[buffer(2)]],  // src_in_tile * rec_count
    device const uint16_t* log_tab      [[buffer(3)]],
    device const uint16_t* alog_tab     [[buffer(4)]],
    constant     uint32_t& stride_words [[buffer(5)]],  // chunksize / 2
    constant     uint32_t& rec_count    [[buffer(6)]],
    constant     uint32_t& src_in_tile  [[buffer(7)]],  // actual blocks this dispatch
    uint2 gid [[thread_position_in_grid]]
) {
    uint word_idx = gid.x;
    uint rec_idx  = gid.y;

    uint16_t acc = 0;

    for (uint ii = 0; ii < src_in_tile; ii++) {
        uint16_t s = src_tile[ii * stride_words + word_idx];
        if (s == 0) continue;

        uint16_t factor = factors_tile[ii * rec_count + rec_idx];
        if (factor == 0) continue;

        uint32_t lc = (uint32_t)log_tab[factor] + (uint32_t)log_tab[s];
        if (lc >= 65535u) lc -= 65535u;
        acc ^= alog_tab[lc];
    }

    if (acc != 0)
        dst[rec_idx * stride_words + word_idx] ^= acc;
}
)MSL";

// ---------------------------------------------------------------------------
// Context struct
// ---------------------------------------------------------------------------
struct MetalRS {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pso;

    id<MTLBuffer> buf_log;           // 65536 × uint16_t — GF16 log table
    id<MTLBuffer> buf_alog;          // 65536 × uint16_t — GF16 antilog table
    id<MTLBuffer> buf_dst;           // recovery_block_count × chunksize (output)
    id<MTLBuffer> buf_src_tile;      // METAL_RS_TILE_SIZE × chunksize (source staging)
    id<MTLBuffer> buf_factors_tile;  // METAL_RS_TILE_SIZE × recovery_block_count × uint16_t

    uint32_t recovery_block_count;
    uint32_t stride_words;   // chunksize / 2
    size_t   chunksize;
    uint32_t tile_fill;      // source blocks currently staged (0 .. METAL_RS_TILE_SIZE)
    bool     ok;
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

MetalRS* metal_rs_create(uint32_t recovery_block_count, size_t chunksize)
{
    MetalRS* ctx = new MetalRS();
    ctx->recovery_block_count = recovery_block_count;
    ctx->chunksize             = chunksize;
    ctx->stride_words          = (uint32_t)(chunksize / 2);
    ctx->tile_fill             = 0;
    ctx->ok                    = false;

    if (recovery_block_count == 0 || chunksize == 0 || chunksize % 2 != 0)
        return ctx;

    ctx->device = MTLCreateSystemDefaultDevice();
    if (!ctx->device) return ctx;

    ctx->queue = [ctx->device newCommandQueue];
    if (!ctx->queue) return ctx;

    NSError *error = nil;
    NSString *src = [NSString stringWithUTF8String:kShaderSrc];
    id<MTLLibrary> lib = [ctx->device newLibraryWithSource:src options:nil error:&error];
    if (!lib) {
        NSLog(@"MetalRS: shader compile error: %@", error);
        return ctx;
    }

    id<MTLFunction> fn = [lib newFunctionWithName:@"rs_process_tiled"];
    ctx->pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
    if (!ctx->pso) {
        NSLog(@"MetalRS: PSO create error: %@", error);
        return ctx;
    }

    uint16_t log_tab[65536], alog_tab[65536];
    build_gf16_tables(log_tab, alog_tab);

    ctx->buf_log = [ctx->device newBufferWithBytes:log_tab
                                            length:sizeof(log_tab)
                                           options:MTLResourceStorageModeShared];
    ctx->buf_alog = [ctx->device newBufferWithBytes:alog_tab
                                             length:sizeof(alog_tab)
                                            options:MTLResourceStorageModeShared];
    ctx->buf_dst = [ctx->device newBufferWithLength:chunksize * recovery_block_count
                                            options:MTLResourceStorageModeShared];
    ctx->buf_src_tile = [ctx->device
                         newBufferWithLength:METAL_RS_TILE_SIZE * chunksize
                                     options:MTLResourceStorageModeShared];
    ctx->buf_factors_tile = [ctx->device
                             newBufferWithLength:METAL_RS_TILE_SIZE * recovery_block_count
                                                * sizeof(uint16_t)
                                         options:MTLResourceStorageModeShared];

    if (!ctx->buf_log || !ctx->buf_alog || !ctx->buf_dst ||
        !ctx->buf_src_tile || !ctx->buf_factors_tile)
        return ctx;

    ctx->ok = true;
    return ctx;
}

void metal_rs_destroy(MetalRS* ctx) { delete ctx; }

bool metal_rs_available(MetalRS* ctx) { return ctx && ctx->ok; }

void metal_rs_clear_output(MetalRS* ctx)
{
    ctx->tile_fill = 0;
    memset(ctx->buf_dst.contents, 0, ctx->chunksize * ctx->recovery_block_count);
}

// Dispatch exactly src_count staged source blocks and wait for completion.
// Called with tile_fill == METAL_RS_TILE_SIZE (full tile) or with the
// remainder at get_output time (partial tile).
static void flush_tile(MetalRS* ctx, uint32_t src_count)
{
    uint32_t word_count = ctx->stride_words;

    id<MTLCommandBuffer>      cb  = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:ctx->buf_src_tile     offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_dst          offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_factors_tile offset:0 atIndex:2];
    [enc setBuffer:ctx->buf_log          offset:0 atIndex:3];
    [enc setBuffer:ctx->buf_alog         offset:0 atIndex:4];
    [enc setBytes:&ctx->stride_words              length:sizeof(uint32_t) atIndex:5];
    [enc setBytes:&ctx->recovery_block_count      length:sizeof(uint32_t) atIndex:6];
    [enc setBytes:&src_count                      length:sizeof(uint32_t) atIndex:7];

    MTLSize gridSize = MTLSizeMake(word_count, ctx->recovery_block_count, 1);
    NSUInteger tgw   = ctx->pso.threadExecutionWidth;
    if (tgw > (NSUInteger)word_count) tgw = word_count;
    MTLSize tgSize = MTLSizeMake(tgw, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
}

void metal_rs_process_block(MetalRS* ctx,
                              const void*     src,
                              size_t          blocklength,
                              const uint16_t* factors,
                              uint32_t        rec_count)
{
    // Copy source block into tile staging buffer at the current slot.
    uint8_t* src_slot = (uint8_t*)ctx->buf_src_tile.contents
                       + ctx->tile_fill * ctx->chunksize;
    memcpy(src_slot, src, blocklength);
    // Zero any padding if blocklength < chunksize (e.g. last block of a file).
    if (blocklength < ctx->chunksize)
        memset(src_slot + blocklength, 0, ctx->chunksize - blocklength);

    // Copy factors for this source block.
    uint16_t* fac_slot = (uint16_t*)ctx->buf_factors_tile.contents
                        + ctx->tile_fill * ctx->recovery_block_count;
    memcpy(fac_slot, factors, rec_count * sizeof(uint16_t));

    ctx->tile_fill++;

    if (ctx->tile_fill == METAL_RS_TILE_SIZE) {
        flush_tile(ctx, METAL_RS_TILE_SIZE);
        ctx->tile_fill = 0;
    }
}

void metal_rs_get_output(MetalRS* ctx, void* cpu_dst)
{
    // Flush any partially-filled tile before reading output.
    if (ctx->tile_fill > 0) {
        flush_tile(ctx, ctx->tile_fill);
        ctx->tile_fill = 0;
    }
    memcpy(cpu_dst, ctx->buf_dst.contents,
           ctx->chunksize * ctx->recovery_block_count);
}

#endif // USE_METAL
