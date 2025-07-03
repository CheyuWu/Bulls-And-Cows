#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>

#include "player.hpp"
#include "utils.hpp"

#define MAX_ROUNDS 20

int main() {
    srand(time(0));

    int gpu_count = 0;
    get_and_validate_gpu_count(gpu_count);

    std::ofstream log_file("game_log.csv");
    if (!log_file.is_open()) {
        perror("Failed to open game_log.csv");
        return 1;
    }
    log_file << "Round,Player,Guess,Feedback\n";

    // If there are two GPUs, we can use both for the game.
    // If only one GPU is available, we will use the same GPU for both players.
    int device0 = 0;
    int device1 = (gpu_count >= 2) ? 1 : 0;
    // FilteringPlayer can change to RandomPlayer for random guessing
    // but FilteringPlayer is more efficient for this game.
    FilteringPlayer gpu1("GPU1-Filter", device0);
    FilteringPlayer gpu2("GPU2-Filter", device1);

    printf("[Game Start]\n");
    printf("%s secret: %s\n", gpu1.name.c_str(), gpu1.secret_code.c_str());
    printf("%s secret: %s\n", gpu2.name.c_str(), gpu2.secret_code.c_str());

    // GPU 1 goes first
    for (int round = 1; round <= MAX_ROUNDS; ++round) {
        printf("\n[Round %d]\n", round);

        std::string g1 = gpu1.make_guess();
        auto [a1, b1] = evaluate_guess(gpu2.secret_code, g1);

        char fb1[8];
        sprintf(fb1, "%dA%dB", a1, b1);
        gpu1.update(g1, fb1);

        printf("%s guesses: %s => %dA%dB\n", gpu1.name.c_str(), g1.c_str(), a1, b1);
        log_file << round << "," << gpu1.name << "," << g1 << "," << fb1 << "\n";

        if (a1 == 4) {
            break;
        }

        std::string g2 = gpu2.make_guess();
        auto [a2, b2] = evaluate_guess(gpu1.secret_code, g2);

        char fb2[8];
        sprintf(fb2, "%dA%dB", a2, b2);
        gpu2.update(g2, fb2);

        printf("%s guesses: %s => %dA%dB\n", gpu2.name.c_str(), g2.c_str(), a2, b2);
        log_file << round << "," << gpu2.name << "," << g2 << "," << fb2 << "\n";

        if (a2 == 4) {
            break;
        }
    }

    log_file.close();
    return 0;
}
