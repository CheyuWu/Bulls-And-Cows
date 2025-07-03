#include "player.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "cuda_utils.cuh"
#include "utils.hpp"

Player::Player(std::string n, int device_id)
    : name(n), device_id(device_id), secret_code(generate_random_code()) {
    possible_codes = get_all_codes();

    printf("Player %s initialized on GPU %d\n", name.c_str(), device_id);

    cudaError_t err = cudaSetDevice(device_id);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to set CUDA device %d: %s\n", device_id, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void Player::update(std::string guess, std::string feedback) {
    history.push_back(guess);
    result.push_back(feedback);
}

RandomPlayer::RandomPlayer(std::string n, int device_id) : Player(n, device_id) {}

std::string RandomPlayer::make_guess() {
    std::random_shuffle(possible_codes.begin(), possible_codes.end());
    return possible_codes[0];
}

FilteringPlayer::FilteringPlayer(std::string n, int device_id) : Player(n, device_id) {}

std::string FilteringPlayer::make_guess() {
    if (history.empty()) {
        return possible_codes[0];
    }

    int total = possible_codes.size();
    char *h_codes = new char[total * 4];

    for (int i = 0; i < total; ++i) {
        memcpy(&h_codes[i * 4], possible_codes[i].c_str(), 4);
    }

    int hist_len = history.size();
    char *h_hist = new char[hist_len * 4];
    int *h_fb = new int[hist_len];

    for (int i = 0; i < hist_len; ++i) {
        memcpy(&h_hist[i * 4], history[i].c_str(), 4);
        int A = result[i][0] - '0', B = result[i][2] - '0';
        h_fb[i] = A * 10 + B;
    }

    char *d_codes, *d_hist;
    int *d_fb;
    bool *d_valid;

    cudaMalloc(&d_codes, total * 4);
    cudaMalloc(&d_hist, hist_len * 4);
    cudaMalloc(&d_fb, hist_len * sizeof(int));
    cudaMalloc(&d_valid, total * sizeof(bool));

    cudaMemcpy(d_codes, h_codes, total * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hist, h_hist, hist_len * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_fb, h_fb, hist_len * sizeof(int), cudaMemcpyHostToDevice);

    launch_filter_candidates_kernel(d_codes, total, d_hist, d_fb, hist_len, d_valid);

    bool *h_valid = new bool[total];
    cudaMemcpy(h_valid, d_valid, total * sizeof(bool), cudaMemcpyDeviceToHost);

    std::vector<std::string> filtered;
    for (int i = 0; i < total; ++i) {
        if (h_valid[i]) {
            filtered.push_back(possible_codes[i]);
        }
    }

    possible_codes = filtered;

    delete[] h_codes;
    delete[] h_hist;
    delete[] h_fb;
    delete[] h_valid;
    cudaFree(d_codes);
    cudaFree(d_hist);
    cudaFree(d_fb);
    cudaFree(d_valid);

    return possible_codes.empty() ? generate_random_code() : possible_codes[0];
}
