#pragma once
#include <string>
#include <vector>

std::string generate_random_code();
std::pair<int, int> evaluate_guess(const std::string &code, const std::string &guess);
std::vector<std::string> generate_all_codes();
void get_and_validate_gpu_count(int &count);

inline std::vector<std::string> &get_all_codes() {
    static std::vector<std::string> codes = generate_all_codes();
    return codes;
}
