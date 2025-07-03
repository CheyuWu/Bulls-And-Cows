#pragma once
#include <string>
#include <vector>

class Player {
   public:
    std::string name;
    int device_id;
    std::string secret_code;
    std::vector<std::string> history;
    std::vector<std::string> result;
    std::vector<std::string> possible_codes;

    Player(std::string n, int device_id);
    virtual std::string make_guess() = 0;
    void update(std::string guess, std::string feedback);
};

class RandomPlayer : public Player {
   public:
    RandomPlayer(std::string n, int device_id);
    std::string make_guess() override;
};

class FilteringPlayer : public Player {
   public:
    FilteringPlayer(std::string n, int device_id);
    std::string make_guess() override;
};
