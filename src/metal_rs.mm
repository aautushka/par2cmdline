// config.h defines USE_METAL (set by configure --enable-metal)
#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#ifdef USE_METAL

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_rs.h"
#include <cstring>

// Reconstruct GF(2^16) log/antilog tables using the same generator as Galois16
// in galois.h (0x1100B). We rebuild here rather than reach into galois.h's
// protected static member.
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

// Metal shader compiled at runtime to avoid adding a separate .metallib
// build step to the autotools chain.
static const char* kShaderSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

kernel void rs_process_galois16(
    device const uint16_t* src          [[buffer(0)]],
    device       uint16_t* dst          [[buffer(1)]],
    device const uint16_t* factors      [[buffer(2)]],
    device const uint16_t* log_tab      [[buffer(3)]],
    device const uint16_t* alog_tab     [[buffer(4)]],
    constant     uint32_t& stride_words [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint word_idx = gid.x;   // position within block (u16 words)
    uint rec_idx  = gid.y;   // recovery block index

    uint16_t factor = factors[rec_idx];
    if (factor == 0) return;

    uint16_t s = src[word_idx];
    if (s == 0) return;

    // GF(2^16) multiply via log/antilog: log[a] + log[b] mod 65535 -> alog
    uint32_t lc = (uint32_t)log_tab[factor] + (uint32_t)log_tab[s];
    if (lc >= 65535u) lc -= 65535u;

    // XOR result into output; stride_words = chunksize/2 matches outputbuffer layout
    dst[rec_idx * stride_words + word_idx] ^= alog_tab[lc];
}
)MSL";

struct MetalRS {
    id<MTLDevice>               device;
    id<MTLCommandQueue>         queue;
    id<MTLComputePipelineState> pso;

    id<MTLBuffer> buf_log;     // 65536 * sizeof(uint16_t)
    id<MTLBuffer> buf_alog;    // 65536 * sizeof(uint16_t)
    id<MTLBuffer> buf_src;     // chunksize bytes
    id<MTLBuffer> buf_dst;     // chunksize * recovery_block_count bytes
    id<MTLBuffer> buf_factors; // recovery_block_count * sizeof(uint16_t)

    uint32_t recovery_block_count;
    uint32_t stride_words;   // chunksize / 2  (u16 words between recovery blocks)
    size_t   chunksize;
    bool     ok;
};

MetalRS* metal_rs_create(uint32_t recovery_block_count, size_t chunksize)
{
    MetalRS* ctx = new MetalRS();
    ctx->recovery_block_count = recovery_block_count;
    ctx->chunksize             = chunksize;
    ctx->stride_words          = (uint32_t)(chunksize / 2);
    ctx->ok                    = false;

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

    id<MTLFunction> fn = [lib newFunctionWithName:@"rs_process_galois16"];
    ctx->pso = [ctx->device newComputePipelineStateWithFunction:fn error:&error];
    if (!ctx->pso) {
        NSLog(@"MetalRS: PSO create error: %@", error);
        return ctx;
    }

    // Upload GF16 tables (constant across all invocations)
    uint16_t log_tab[65536], alog_tab[65536];
    build_gf16_tables(log_tab, alog_tab);

    ctx->buf_log = [ctx->device newBufferWithBytes:log_tab
                                            length:sizeof(log_tab)
                                           options:MTLResourceStorageModeShared];
    ctx->buf_alog = [ctx->device newBufferWithBytes:alog_tab
                                             length:sizeof(alog_tab)
                                            options:MTLResourceStorageModeShared];

    ctx->buf_src = [ctx->device newBufferWithLength:chunksize
                                            options:MTLResourceStorageModeShared];

    ctx->buf_dst = [ctx->device newBufferWithLength:chunksize * recovery_block_count
                                            options:MTLResourceStorageModeShared];

    ctx->buf_factors = [ctx->device newBufferWithLength:recovery_block_count * sizeof(uint16_t)
                                               options:MTLResourceStorageModeShared];

    ctx->ok = true;
    return ctx;
}

void metal_rs_destroy(MetalRS* ctx) { delete ctx; }

bool metal_rs_available(MetalRS* ctx) { return ctx && ctx->ok; }

void metal_rs_clear_output(MetalRS* ctx)
{
    memset(ctx->buf_dst.contents, 0, ctx->chunksize * ctx->recovery_block_count);
}

void metal_rs_process_block(MetalRS* ctx,
                              const void*     src,
                              size_t          blocklength,
                              const uint16_t* factors,
                              uint32_t        rec_count)
{
    uint32_t word_count = (uint32_t)(blocklength / 2);
    if (word_count == 0) return;

    // Upload source block and factors
    memcpy(ctx->buf_src.contents, src, blocklength);
    memcpy(ctx->buf_factors.contents, factors, rec_count * sizeof(uint16_t));

    id<MTLCommandBuffer> cb  = [ctx->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

    [enc setComputePipelineState:ctx->pso];
    [enc setBuffer:ctx->buf_src     offset:0 atIndex:0];
    [enc setBuffer:ctx->buf_dst     offset:0 atIndex:1];
    [enc setBuffer:ctx->buf_factors offset:0 atIndex:2];
    [enc setBuffer:ctx->buf_log     offset:0 atIndex:3];
    [enc setBuffer:ctx->buf_alog    offset:0 atIndex:4];
    // stride_words as a small inline constant (no heap allocation needed)
    [enc setBytes:&ctx->stride_words length:sizeof(uint32_t) atIndex:5];

    // One thread per (word, recovery_block) pair.
    // dispatchThreads: dispatches exactly gridSize threads — no bounds check needed.
    MTLSize gridSize = MTLSizeMake(word_count, rec_count, 1);
    NSUInteger tgw   = ctx->pso.threadExecutionWidth;   // typically 32 on Apple Silicon
    if (tgw > (NSUInteger)word_count) tgw = word_count;
    MTLSize tgSize = MTLSizeMake(tgw, 1, 1);

    [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
    [enc endEncoding];

    [cb commit];
    [cb waitUntilCompleted];  // synchronous: caller can reuse inputbuffer immediately
}

void metal_rs_get_output(MetalRS* ctx, void* cpu_dst)
{
    memcpy(cpu_dst, ctx->buf_dst.contents,
           ctx->chunksize * ctx->recovery_block_count);
}

#endif // USE_METAL
