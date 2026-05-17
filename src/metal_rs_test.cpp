#ifdef HAVE_CONFIG_H
#  include <config.h>
#endif

#ifdef USE_METAL

// End-to-end byte comparison: CPU (ReedSolomon::Process) vs Metal GPU path.
// Generates random source data, computes recovery blocks both ways, asserts equality.

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "libpar2internal.h"
#include "reedsolomon.h"
#include "metal_rs.h"

static const int NUM_SRC = 20;
static const int NUM_REC = 5;
static const int BUF_SIZE = 4096;   // must be a multiple of 4 (Galois16 processes 4 bytes at a time)

static int run_test(unsigned int seed, int low_exp)
{
    // ---- build input data ------------------------------------------------
    srand(seed);
    u8 src[NUM_SRC][BUF_SIZE];
    for (int i = 0; i < NUM_SRC; i++)
        for (int k = 0; k < BUF_SIZE; k++)
            src[i][k] = (u8)(rand() % 256);

    // ---- CPU recovery output ---------------------------------------------
    u8 cpu_out[NUM_REC][BUF_SIZE];
    memset(cpu_out, 0, sizeof(cpu_out));

    ReedSolomon<Galois16> rs;
    if (!rs.SetInput(NUM_SRC, std::cout, std::cerr)) { std::cerr << "SetInput failed\n"; return 1; }
    if (!rs.SetOutput(false, (u16)low_exp, (u16)(low_exp + NUM_REC - 1))) { std::cerr << "SetOutput failed\n"; return 1; }
    if (!rs.Compute(nlSilent, std::cout, std::cerr)) { std::cerr << "Compute failed\n"; return 1; }

    for (int i = 0; i < NUM_SRC; i++)
        for (int j = 0; j < NUM_REC; j++)
            rs.Process(BUF_SIZE, (u32)i, src[i], (u32)j, cpu_out[j]);

    // ---- GPU recovery output ---------------------------------------------
    // chunksize = BUF_SIZE (single pass), stride_words = BUF_SIZE/2
    MetalRS* ctx = metal_rs_create(NUM_REC, BUF_SIZE);
    if (!metal_rs_available(ctx)) {
        std::cerr << "SKIP: No Metal device available.\n";
        metal_rs_destroy(ctx);
        return 0;
    }

    metal_rs_clear_output(ctx);

    for (int i = 0; i < NUM_SRC; i++) {
        std::vector<uint16_t> factors(NUM_REC);
        for (int j = 0; j < NUM_REC; j++)
            factors[j] = (uint16_t)rs.GetFactor((u32)j, (u32)i);
        metal_rs_process_block(ctx, src[i], BUF_SIZE, factors.data(), NUM_REC);
    }

    // gpu_out layout: rec j at offset j * BUF_SIZE  (stride = chunksize = BUF_SIZE)
    u8 gpu_flat[NUM_REC * BUF_SIZE];
    metal_rs_get_output(ctx, gpu_flat);
    metal_rs_destroy(ctx);

    // ---- byte-for-byte comparison ----------------------------------------
    for (int j = 0; j < NUM_REC; j++) {
        const u8* gpu_rec = &gpu_flat[j * BUF_SIZE];
        for (int k = 0; k < BUF_SIZE; k++) {
            if (cpu_out[j][k] != gpu_rec[k]) {
                std::cerr << "MISMATCH seed=" << seed
                          << " low_exp=" << low_exp
                          << " rec=" << j << " byte=" << k
                          << " cpu=" << (int)cpu_out[j][k]
                          << " gpu=" << (int)gpu_rec[k] << "\n";
                return 1;
            }
        }
    }
    return 0;
}

int main()
{
    // Several seeds and exponent offsets to exercise different matrix coefficients
    struct { unsigned int seed; int low_exp; } cases[] = {
        { 12345,  0 },
        { 98765,  0 },
        { 11111, 10 },
        { 22222,  1 },
    };

    for (auto& c : cases) {
        if (run_test(c.seed, c.low_exp)) {
            std::cerr << "FAILED: metal_rs_test (seed=" << c.seed << " low_exp=" << c.low_exp << ")\n";
            return 1;
        }
        std::cout << "  [PASS] seed=" << c.seed << " low_exp=" << c.low_exp << "\n";
    }
    std::cout << "metal_rs_test: all cases passed.\n";
    return 0;
}

#else // USE_METAL

#include <iostream>
int main() {
    std::cout << "metal_rs_test: skipped (USE_METAL not defined).\n";
    return 0;
}

#endif // USE_METAL
