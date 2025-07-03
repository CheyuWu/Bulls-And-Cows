#pragma once
#define CODE_LEN 4

__device__ void evaluate(const char *code, const char *guess, int &A, int &B);
__global__ void filter_candidates_kernel(char *d_codes, int total_codes, char *d_history,
                                         int *d_feedbacks, int history_len, bool *d_valid);

void launch_filter_candidates_kernel(char *d_codes, int total_codes, char *d_history,
                                     int *d_feedbacks, int history_len, bool *d_valid);
