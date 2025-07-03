#include "cuda_utils.cuh"

__device__ void evaluate(const char *code, const char *guess, int &A, int &B) {
    A = 0;
    B = 0;
    for (int i = 0; i < CODE_LEN; ++i) {
        if (guess[i] == code[i]) {
            A++;
        } else {
            for (int j = 0; j < CODE_LEN; ++j) {
                if (guess[i] == code[j]) {
                    B++;
                    break;
                }
            }
        }
    }
}

__global__ void filter_candidates_kernel(char *d_codes, int total_codes, char *d_history,
                                         int *d_feedbacks, int history_len, bool *d_valid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_codes) {
        return;
    }

    bool valid = true;
    for (int h = 0; h < history_len; ++h) {
        int A, B;
        evaluate(&d_codes[idx * CODE_LEN], &d_history[h * CODE_LEN], A, B);

        if ((A * 10 + B) != d_feedbacks[h]) {
            valid = false;
            break;
        }
    }
    d_valid[idx] = valid;
}

void launch_filter_candidates_kernel(char *d_codes, int total_codes, char *d_history,
                                     int *d_feedbacks, int history_len, bool *d_valid) {
    int blockSize = 256;
    int gridSize = (total_codes + blockSize - 1) / blockSize;
    filter_candidates_kernel<<<gridSize, blockSize>>>(d_codes, total_codes, d_history, d_feedbacks,
                                                      history_len, d_valid);

    cudaDeviceSynchronize();
}
