#include "utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

std::string generate_random_code() {
    std::string digits = "0123456789";
    std::random_shuffle(digits.begin(), digits.end());
    return digits.substr(0, 4);
}

std::pair<int, int> evaluate_guess(const std::string &code, const std::string &guess) {
    int A = 0, B = 0;
    for (int i = 0; i < 4; i++) {
        if (guess[i] == code[i]) {
            A++;
        } else if (code.find(guess[i]) != std::string::npos) {
            B++;
        }
    }
    return {A, B};
}
std::vector<std::string> generate_all_codes() {
    std::vector<std::string> codes;
    for (char a = '0'; a <= '9'; ++a) {
        for (char b = '0'; b <= '9'; ++b) {
            if (b == a) {
                continue;
            }
            for (char c = '0'; c <= '9'; ++c) {
                if (c == a || c == b) {
                    continue;
                }
                for (char d = '0'; d <= '9'; ++d) {
                    if (d == a || d == b || d == c) {
                        continue;
                    }
                    std::string code;
                    code += a;
                    code += b;
                    code += c;
                    code += d;
                    codes.push_back(code);
                }
            }
        }
    }
    return codes;
}

void get_and_validate_gpu_count(int &count) {
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error getting device count: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    if (count == 0) {
        fprintf(stderr, "No CUDA devices found.\n");
        exit(EXIT_FAILURE);
    }
    printf("Number of CUDA devices: %d\n", count);
}
