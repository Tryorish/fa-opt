// Compile on CPU: c++ -std=c++17 test_lds_layout.cpp -o /tmp/test_lds_layout
#include "shared_layout.h"
#include <algorithm>
#include <iostream>
#include <set>

template <int Stride, bool V = false>
bool check_vectors_and_bijection(int rows) {
    std::set<int> offsets;
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < Stride; ++col) {
            const int offset = V ? lds_v<Stride>(row, col) : lds<Stride>(row, col);
            if (offset < 0 || offset >= rows * Stride || !offsets.insert(offset).second)
                return false;
            const int base = V ? lds_v<Stride>(row, col / 4 * 4) : lds<Stride>(row, col / 4 * 4);
            if (base % 4 || offset != base + col % 4)
                return false;
        }
    }
    return true;
}

int max_words_per_bank(const std::set<int> (&banks)[32]) {
    int worst = 0;
    for (const auto& words : banks)
        worst = std::max(worst, static_cast<int>(words.size()));
    return worst;
}

bool check_matrix_v() {
    // The matrix instruction reads 8 halves/request lane, then redistributes
    // them. These phases are the reference model; target tags verify gfx936.
    constexpr int phases[8][8] = {
        {0,4,1,5,18,22,19,23}, {2,6,3,7,16,20,17,21},
        {8,12,9,13,26,30,27,31}, {10,14,11,15,24,28,25,29},
        {32,36,33,37,50,54,51,55}, {34,38,35,39,48,52,49,53},
        {40,44,41,45,58,62,59,63}, {42,46,43,47,56,60,57,61}};
    std::set<int> offsets;
    for (int row = 0; row < 128; ++row) {
        for (int col = 0; col < 64; ++col) {
            const int offset = lds_v_matrix(row, col);
            if (offset < 0 || offset >= 128 * 64 || !offsets.insert(offset).second ||
                offset != lds_v_matrix(row, col / 8 * 8) + col % 8)
                return false;
        }
        // One 128B store phase covers all 64 halves of this row.
        std::set<int> banks[32];
        for (int col = 0; col < 64; col += 2) {
            const int word = lds_v_matrix(row, col) / 2;
            banks[word % 32].insert(word);
        }
        if (max_words_per_bank(banks) != 1) return false;
    }
    for (int k = 0; k < 128; k += 16) {
        for (int panel = 0; panel < 64; panel += 32) {
            for (const auto& phase : phases) {
                std::set<int> banks[32];
                for (int lane : phase) {
                    const int base = lds_v_matrix(k + lane / 4, panel + (lane % 4) * 8);
                    if (base % 8) return false;
                    for (int r = 0; r < 8; r += 2) {
                        const int word = (base + r) / 2;
                        banks[word % 32].insert(word);
                    }
                }
                if (max_words_per_bank(banks) != 1) return false;
            }
        }
    }
    return true;
}

bool check_async_v() {
    // The instruction deposits 8 halves at lane*8 in each wave segment.
    // Check production source indices reconstruct the SAME logical V tile.
    for (int keys : {32, 64, 128}) {
        std::set<int> sources;
        for (int chunk = 0; chunk < keys / 32; ++chunk) {
            for (int tid = 0; tid < 256; ++tid) {
                const int source = async_v_source(tid, chunk);
                if (source % 8) return false;
                for (int r = 0; r < 8; ++r) {
                    const int i = source + r;
                    const int dest = (tid + chunk * 256) * 8 + r;
                    if (i < 0 || i >= keys * 64 || !sources.insert(i).second ||
                        lds_v_matrix(i / 64, i % 64) != dest)
                        return false;
                }
            }
        }
        if (sources.size() != static_cast<size_t>(keys * 64)) return false;
    }
    return true;
}

bool check_async_k() {
    for (int keys : {32, 64, 128}) {
        std::set<int> sources;
        for (int chunk = 0; chunk < keys / 16; ++chunk) {
            for (int tid = 0; tid < 256; ++tid) {
                const int source = async_k_source(tid, chunk);
                if (source % 4) return false;
                for (int r = 0; r < 4; ++r) {
                    const int i = source + r;
                    const int dest = (tid + chunk * 256) * 4 + r;
                    if (i < 0 || i >= keys * 64 || !sources.insert(i).second ||
                        lds<64>(i / 64, i % 64) != dest)
                        return false;
                }
            }
        }
        if (sources.size() != static_cast<size_t>(keys * 64)) return false;
    }
    return true;
}

template <int Keys>
bool check_v_banks() {
    int worst_write = 0, worst_read = 0;
    // Same 32-thread chunks and scalar stores as stage_kv; count DISTINCT
    // dwords, so writes to different halves of one dword aren't double counted.
    for (int base = 0; base < Keys * 8; base += 32) {
        for (int r = 0; r < 8; ++r) {
            std::set<int> banks[32];
            for (int t = base; t < base + 32; ++t) {
                const int word = lds_v<Keys>((t % 8) * 8 + r, t / 8) / 2;
                banks[word % 32].insert(word);
            }
            worst_write = std::max(worst_write, max_words_per_bank(banks));
        }
    }
    // Four-half (64-bit) PV reads, each 16-lane group addresses 32 dwords.
    for (int d = 0; d < 4; ++d) {
        for (int key = 0; key < Keys; key += 4) {
            std::set<int> banks[32];
            for (int lane = 0; lane < 16; ++lane) {
                for (int r : {0, 2}) {
                    const int word = lds_v<Keys>(d * 16 + lane, key + r) / 2;
                    banks[word % 32].insert(word);
                }
            }
            worst_read = std::max(worst_read, max_words_per_bank(banks));
        }
    }
    std::cout << "KV tile=" << Keys << ", max distinct words/bank: V stores=" << worst_write
              << ", V reads=" << worst_read << '\n';
    return worst_write == 1 && worst_read == 1;
}

int main() {
    if (!check_vectors_and_bijection<64>(128) ||
        !check_vectors_and_bijection<32, true>(64) ||
        !check_vectors_and_bijection<64, true>(64))
        return 1;
    const bool good32 = check_v_banks<32>();
    const bool good64 = check_v_banks<64>();
    return good32 && good64 && check_matrix_v() && check_async_v() && check_async_k() ? 0 : 1;
}
