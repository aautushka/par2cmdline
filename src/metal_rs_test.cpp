#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#ifdef USE_METAL

// End-to-end correctness suite: CPU (ReedSolomon::Process) vs Metal tiled GPU path.
//
// Coverage goals:
//   Tile boundaries  — sub-tile, exact tile, multi-tile, odd remainders
//   Recovery count   — 1, 2, mid-range
//   Block size       — minimum (4 B), small, large
//   Exponent offsets — various low_exp values affect the RS factor matrix
//   Zero inputs      — all-zero source → all-zero output
//   State isolation  — clear_output resets accumulation; second pass is clean
//   Reuse            — same data processed twice gives identical results

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>

#include "libpar2internal.h"
#include "reedsolomon.h"
#include "metal_rs.h"

static const int T = (int)METAL_RS_TILE_SIZE;   // shorthand for boundary arithmetic

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool gpu_skipped = false;  // set true if no Metal device; suppresses further noise

// Run one parameterised CPU-vs-GPU comparison.
// Returns 0 on pass, 1 on fail, -1 on skip (no GPU).
static int run_test(unsigned int seed,
                    int num_src, int num_rec, int buf_size, int low_exp,
                    const char* label)
{
    // ---- source data -------------------------------------------------------
    srand(seed);
    std::vector<std::vector<u8>> src(num_src, std::vector<u8>(buf_size));
    for (auto& blk : src)
        for (auto& b : blk) b = (u8)(rand() % 256);

    // ---- CPU reference output ----------------------------------------------
    std::vector<std::vector<u8>> cpu_out(num_rec, std::vector<u8>(buf_size, 0));

    ReedSolomon<Galois16> rs;
    if (!rs.SetInput(num_src, std::cout, std::cerr))
        { std::cerr << label << ": SetInput failed\n"; return 1; }
    if (!rs.SetOutput(false, (u16)low_exp, (u16)(low_exp + num_rec - 1)))
        { std::cerr << label << ": SetOutput failed\n"; return 1; }
    if (!rs.Compute(nlSilent, std::cout, std::cerr))
        { std::cerr << label << ": Compute failed\n"; return 1; }

    for (int i = 0; i < num_src; i++)
        for (int j = 0; j < num_rec; j++)
            rs.Process(buf_size, (u32)i, src[i].data(), (u32)j, cpu_out[j].data());

    // ---- GPU output --------------------------------------------------------
    MetalRS* ctx = metal_rs_create((u32)num_rec, (size_t)buf_size);
    if (!metal_rs_available(ctx)) {
        if (!gpu_skipped) { std::cerr << "  SKIP: no Metal device\n"; gpu_skipped = true; }
        metal_rs_destroy(ctx);
        return -1;
    }

    metal_rs_clear_output(ctx);
    for (int i = 0; i < num_src; i++) {
        std::vector<uint16_t> factors(num_rec);
        for (int j = 0; j < num_rec; j++)
            factors[j] = (uint16_t)rs.GetFactor((u32)j, (u32)i);
        metal_rs_process_block(ctx, src[i].data(), buf_size, factors.data(), num_rec);
    }

    std::vector<u8> gpu_flat((size_t)num_rec * buf_size);
    metal_rs_get_output(ctx, gpu_flat.data());
    metal_rs_destroy(ctx);

    // ---- byte-for-byte comparison ------------------------------------------
    for (int j = 0; j < num_rec; j++) {
        const u8* gpu_rec = gpu_flat.data() + j * buf_size;
        for (int k = 0; k < buf_size; k++) {
            if (cpu_out[j][k] != gpu_rec[k]) {
                std::cerr << "MISMATCH " << label
                          << "  rec=" << j << " byte=" << k
                          << "  cpu=" << (int)cpu_out[j][k]
                          << "  gpu=" << (int)gpu_rec[k] << "\n";
                return 1;
            }
        }
    }
    return 0;
}

// Verify output is all zeros when no process_block calls are made.
static int test_empty_output()
{
    const int NUM_REC = 5, BUF = 4096;
    MetalRS* ctx = metal_rs_create(NUM_REC, BUF);
    if (!metal_rs_available(ctx)) { metal_rs_destroy(ctx); return -1; }

    metal_rs_clear_output(ctx);
    // Intentionally call get_output without any process_block.
    std::vector<u8> out((size_t)NUM_REC * BUF, 0xFF);
    metal_rs_get_output(ctx, out.data());
    metal_rs_destroy(ctx);

    for (auto b : out)
        if (b != 0) { std::cerr << "FAIL test_empty_output: non-zero byte in empty output\n"; return 1; }
    return 0;
}

// Verify that all-zero source data produces all-zero output (any factor × 0 = 0).
// Uses T+1 blocks so the path crosses a tile boundary.
static int test_zero_source()
{
    const int NUM_SRC = T + 1, NUM_REC = 5, BUF = 4096;
    std::vector<u8> zero_src(BUF, 0);
    std::vector<uint16_t> factors(NUM_REC, 1);  // factor value doesn't matter; src is zero

    MetalRS* ctx = metal_rs_create(NUM_REC, BUF);
    if (!metal_rs_available(ctx)) { metal_rs_destroy(ctx); return -1; }

    metal_rs_clear_output(ctx);
    for (int i = 0; i < NUM_SRC; i++)
        metal_rs_process_block(ctx, zero_src.data(), BUF, factors.data(), NUM_REC);

    std::vector<u8> out((size_t)NUM_REC * BUF, 0xFF);
    metal_rs_get_output(ctx, out.data());
    metal_rs_destroy(ctx);

    for (auto b : out)
        if (b != 0) { std::cerr << "FAIL test_zero_source: non-zero byte\n"; return 1; }
    return 0;
}

// Verify clear_output + second pass is identical to a fresh pass with the same data.
// Tests that no residue from the first pass bleeds into the second.
static int test_reuse()
{
    const int NUM_SRC = T + 3, NUM_REC = 4, BUF = 4096;

    srand(55555);
    std::vector<std::vector<u8>> src(NUM_SRC, std::vector<u8>(BUF));
    for (auto& blk : src)
        for (auto& b : blk) b = (u8)(rand() % 256);

    ReedSolomon<Galois16> rs;
    if (!rs.SetInput(NUM_SRC, std::cout, std::cerr)) return 1;
    if (!rs.SetOutput(false, 0, NUM_REC - 1))        return 1;
    if (!rs.Compute(nlSilent, std::cout, std::cerr)) return 1;

    std::vector<std::vector<uint16_t>> all_factors(NUM_SRC, std::vector<uint16_t>(NUM_REC));
    for (int i = 0; i < NUM_SRC; i++)
        for (int j = 0; j < NUM_REC; j++)
            all_factors[i][j] = (uint16_t)rs.GetFactor((u32)j, (u32)i);

    auto do_pass = [&]() -> std::vector<u8> {
        MetalRS* ctx = metal_rs_create(NUM_REC, BUF);
        if (!metal_rs_available(ctx)) { metal_rs_destroy(ctx); return {}; }
        metal_rs_clear_output(ctx);
        for (int i = 0; i < NUM_SRC; i++)
            metal_rs_process_block(ctx, src[i].data(), BUF,
                                   all_factors[i].data(), NUM_REC);
        std::vector<u8> out((size_t)NUM_REC * BUF);
        metal_rs_get_output(ctx, out.data());
        metal_rs_destroy(ctx);
        return out;
    };

    auto out1 = do_pass();
    if (out1.empty()) return -1;
    auto out2 = do_pass();
    if (out2.empty()) return -1;

    if (out1 != out2) {
        std::cerr << "FAIL test_reuse: second pass differs from first\n";
        return 1;
    }
    return 0;
}

// Verify that after clear_output, processing a DIFFERENT dataset gives the
// correct result for that dataset — not contaminated by the prior run.
static int test_isolation()
{
    const int NUM_SRC = T + 5, NUM_REC = 3, BUF = 4096;

    // Dataset A
    srand(11111);
    std::vector<std::vector<u8>> srcA(NUM_SRC, std::vector<u8>(BUF));
    for (auto& blk : srcA)
        for (auto& b : blk) b = (u8)(rand() % 256);

    // Dataset B — different seed
    srand(22222);
    std::vector<std::vector<u8>> srcB(NUM_SRC, std::vector<u8>(BUF));
    for (auto& blk : srcB)
        for (auto& b : blk) b = (u8)(rand() % 256);

    ReedSolomon<Galois16> rs;
    if (!rs.SetInput(NUM_SRC, std::cout, std::cerr)) return 1;
    if (!rs.SetOutput(false, 0, NUM_REC - 1))        return 1;
    if (!rs.Compute(nlSilent, std::cout, std::cerr)) return 1;

    std::vector<std::vector<uint16_t>> factors(NUM_SRC, std::vector<uint16_t>(NUM_REC));
    for (int i = 0; i < NUM_SRC; i++)
        for (int j = 0; j < NUM_REC; j++)
            factors[i][j] = (uint16_t)rs.GetFactor((u32)j, (u32)i);

    // CPU reference for dataset B only
    std::vector<std::vector<u8>> cpu_outB(NUM_REC, std::vector<u8>(BUF, 0));
    for (int i = 0; i < NUM_SRC; i++)
        for (int j = 0; j < NUM_REC; j++)
            rs.Process(BUF, (u32)i, srcB[i].data(), (u32)j, cpu_outB[j].data());

    MetalRS* ctx = metal_rs_create(NUM_REC, BUF);
    if (!metal_rs_available(ctx)) { metal_rs_destroy(ctx); return -1; }

    // Pass A (discarded)
    metal_rs_clear_output(ctx);
    for (int i = 0; i < NUM_SRC; i++)
        metal_rs_process_block(ctx, srcA[i].data(), BUF, factors[i].data(), NUM_REC);
    { std::vector<u8> discard((size_t)NUM_REC * BUF); metal_rs_get_output(ctx, discard.data()); }

    // Pass B
    metal_rs_clear_output(ctx);
    for (int i = 0; i < NUM_SRC; i++)
        metal_rs_process_block(ctx, srcB[i].data(), BUF, factors[i].data(), NUM_REC);
    std::vector<u8> gpu_outB((size_t)NUM_REC * BUF);
    metal_rs_get_output(ctx, gpu_outB.data());
    metal_rs_destroy(ctx);

    for (int j = 0; j < NUM_REC; j++) {
        const u8* gpu = gpu_outB.data() + j * BUF;
        for (int k = 0; k < BUF; k++) {
            if (cpu_outB[j][k] != gpu[k]) {
                std::cerr << "FAIL test_isolation: rec=" << j << " byte=" << k << "\n";
                return 1;
            }
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main()
{
    int passed = 0, failed = 0, skipped = 0;

    auto run = [&](int rc, const char* label) {
        if (rc == 0) { std::cout << "  [PASS] " << label << "\n"; passed++; }
        else if (rc < 0) { skipped++; }
        else { std::cerr << "  [FAIL] " << label << "\n"; failed++; }
    };

    // -----------------------------------------------------------------------
    // Original cases (backward compatibility with pre-tiling baseline)
    // -----------------------------------------------------------------------
    run(run_test(12345, 20, 5, 4096,  0, "orig seed=12345 low_exp=0"),
        "orig seed=12345 low_exp=0");
    run(run_test(98765, 20, 5, 4096,  0, "orig seed=98765 low_exp=0"),
        "orig seed=98765 low_exp=0");
    run(run_test(11111, 20, 5, 4096, 10, "orig seed=11111 low_exp=10"),
        "orig seed=11111 low_exp=10");
    run(run_test(22222, 20, 5, 4096,  1, "orig seed=22222 low_exp=1"),
        "orig seed=22222 low_exp=1");

    // -----------------------------------------------------------------------
    // Tile boundary cases — exercise every branch in the flush logic
    // -----------------------------------------------------------------------

    // Strictly sub-tile (tile never filled; everything flushed at get_output)
    run(run_test(42,  1, 5, 4096, 0, "tile: num_src=1 (single block)"),
        "tile: num_src=1 (single block)");
    run(run_test(42,  2, 5, 4096, 0, "tile: num_src=2"),
        "tile: num_src=2");
    run(run_test(42, T-1, 5, 4096, 0, "tile: num_src=T-1 (just under full tile)"),
        "tile: num_src=T-1 (just under full tile)");

    // Exactly one tile (flush at fill, empty get_output)
    run(run_test(42, T, 5, 4096, 0, "tile: num_src=T (exactly one tile)"),
        "tile: num_src=T (exactly one tile)");

    // One full tile + partial remainder
    run(run_test(42, T+1,   5, 4096, 0, "tile: num_src=T+1"),
        "tile: num_src=T+1");
    run(run_test(42, T+T/2, 5, 4096, 0, "tile: num_src=T+T/2"),
        "tile: num_src=T+T/2");

    // Exactly two tiles (two fill-flushes, empty get_output)
    run(run_test(42, 2*T-1, 5, 4096, 0, "tile: num_src=2T-1"),
        "tile: num_src=2T-1");
    run(run_test(42, 2*T,   5, 4096, 0, "tile: num_src=2T (two exact tiles)"),
        "tile: num_src=2T (two exact tiles)");
    run(run_test(42, 2*T+1, 5, 4096, 0, "tile: num_src=2T+1"),
        "tile: num_src=2T+1");

    // Three tiles + odd partial
    run(run_test(99, 3*T,    7, 4096, 3, "tile: num_src=3T"),
        "tile: num_src=3T");
    run(run_test(99, 3*T+17, 7, 4096, 3, "tile: num_src=3T+17"),
        "tile: num_src=3T+17");

    // -----------------------------------------------------------------------
    // Recovery block count edge cases
    // -----------------------------------------------------------------------
    run(run_test(42, 20,  1, 4096, 0, "rec: num_rec=1 (single recovery block)"),
        "rec: num_rec=1 (single recovery block)");
    run(run_test(42, 20,  2, 4096, 0, "rec: num_rec=2"),
        "rec: num_rec=2");
    run(run_test(42, 20, T,  4096, 0, "rec: num_rec=T (same as tile size)"),
        "rec: num_rec=T (same as tile size)");

    // -----------------------------------------------------------------------
    // Block size edge cases
    //   buf_size must be a multiple of 4 (Galois16 processes u32 at a time)
    // -----------------------------------------------------------------------
    run(run_test(42, 20, 5,     4, 0, "buf: buf_size=4 (minimum, 2 u16 words)"),
        "buf: buf_size=4 (minimum, 2 u16 words)");
    run(run_test(42, 20, 5,     8, 0, "buf: buf_size=8"),
        "buf: buf_size=8");
    run(run_test(42, 20, 5, 16384, 0, "buf: buf_size=16384 (16 KB)"),
        "buf: buf_size=16384 (16 KB)");
    run(run_test(42, 20, 5, 65536, 0, "buf: buf_size=65536 (64 KB)"),
        "buf: buf_size=65536 (64 KB)");

    // -----------------------------------------------------------------------
    // Exponent offset — shifts which GF elements appear as factors
    // -----------------------------------------------------------------------
    run(run_test(42, 20, 5, 4096,   0, "exp: low_exp=0"),   "exp: low_exp=0");
    run(run_test(42, 20, 5, 4096,   1, "exp: low_exp=1"),   "exp: low_exp=1");
    run(run_test(42, 20, 5, 4096,  50, "exp: low_exp=50"),  "exp: low_exp=50");
    run(run_test(42, 20, 5, 4096, 100, "exp: low_exp=100"), "exp: low_exp=100");

    // -----------------------------------------------------------------------
    // Compound: tile boundary + small block + high exponent
    // -----------------------------------------------------------------------
    run(run_test(77, T+3, 3, 4, 10, "compound: T+3 src, buf=4, low_exp=10"),
        "compound: T+3 src, buf=4, low_exp=10");
    run(run_test(77, 2*T+1, 1, 8, 5, "compound: 2T+1 src, rec=1, buf=8, low_exp=5"),
        "compound: 2T+1 src, rec=1, buf=8, low_exp=5");

    // -----------------------------------------------------------------------
    // State management
    // -----------------------------------------------------------------------
    run(test_empty_output(),  "state: empty output (no process_block calls)");
    run(test_zero_source(),   "state: all-zero source → all-zero output");
    run(test_reuse(),         "state: clear + reprocess same data → identical output");
    run(test_isolation(),     "state: clear + different data → no contamination from prior pass");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\nmetal_rs_test: "
              << passed  << " passed, "
              << failed  << " failed, "
              << skipped << " skipped.\n";
    return failed > 0 ? 1 : 0;
}

#else // USE_METAL

#include <iostream>
int main() {
    std::cout << "metal_rs_test: skipped (USE_METAL not defined).\n";
    return 0;
}

#endif // USE_METAL
