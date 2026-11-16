// CPU proof of complete D128 copies and all four matrix-read panels.
#include "shared_layout.h"
#include <iostream>
#include <vector>

int main() {
    constexpr int D = 128;
    for (int keys : {32, 64}) {
        for (int width : {4, 8}) {
            std::vector<int> shared(keys * D, -1);
            for (int chunk = 0; chunk < keys * D / (256 * width); ++chunk) {
                for (int tid = 0; tid < 256; ++tid) {
                    const int src = width == 4 ? async_k_source(tid, chunk)
                                               : async_v_source(tid, chunk);
                    if (src < 0 || src + width > keys * D || src % width) return 1;
                    for (int r = 0; r < width; ++r)
                        shared[(chunk * 256 + tid) * width + r] = src + r;
                }
            }
            for (int row = 0; row < keys; ++row) {
                for (int col = 0; col < D; ++col) {
                    const int addr = width == 4 ? lds<D>(row, col) : lds_v_matrix(row, col);
                    if (addr < 0 || addr >= keys * D || shared[addr] != row * D + col) {
                        std::cerr << "copy mismatch: width=" << width << " row=" << row
                                  << " col=" << col << '\n';
                        return 1;
                    }
                }
            }
        }
        // Every matrix request must read eight consecutive logical channels.
        for (int key = 0; key < keys; key += 16) {
            for (int panel = 0; panel < 4; ++panel) {
                std::vector<bool> seen(16 * 32);
                for (int lane = 0; lane < 64; ++lane) {
                    int row = key + lane / 4, col = panel * 32 + (lane % 4) * 8;
                    int base = lds_v_matrix(row, col);
                    for (int r = 0; r < 8; ++r) {
                        if (base % 8 || base + r != lds_v_matrix(row, col + r) ||
                            (base + r) / D != row) return 1;
                        seen[(row - key) * 32 + col % 32 + r] = true;
                    }
                }
                for (bool covered : seen) if (!covered) return 1;
            }
        }
    }
    std::cout << "D128 K/V copies and four matrix panels match\n";
}
