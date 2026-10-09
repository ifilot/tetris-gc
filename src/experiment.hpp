#pragma once
#include "tetris.hpp"
#include <string>

namespace tetris {
enum class FileKind : std::uint32_t { training = 1, endurance = 2, statistics = 3 };
struct Config {
    Rules rules{Rules::article};
    std::uint64_t seed{1}, pieces{500}, validation_pieces{500}, checkpoint_pieces{100000};
    std::uint32_t population{1000}, games{100}, generations{20}, runs{1}, validation_games{100};
    double mutation{0.05};
    std::uint32_t sampling{}; // 0 common, 1 independent candidate streams, 2 refreshed common streams.
};
struct History {
    std::uint32_t generation{};
    std::uint64_t best{}, evaluated_pieces{};
    double mean{};
    Weights weights{};
};
struct TrainingRun {
    std::uint64_t seed{}, random_state{}, next_id{};
    std::int32_t generation{-1};
    std::vector<Candidate> population;
    std::vector<History> history;
    std::vector<GameState> validation; // Winner games followed by paired baseline games.
};
struct Policy {
    std::uint64_t training_seed{};
    Weights weights{};
    bool baseline{};
};
struct Experiment {
    FileKind kind{FileKind::training};
    Config config;
    std::vector<TrainingRun> training;
    std::vector<Policy> policies;
    std::vector<GameState> games; // policy-major order, config.games per policy.
};
void save_binary(const std::string& path, const Experiment& experiment);
Experiment load_binary(const std::string& path);
struct Summary {
    std::uint64_t games{}, topouts{}, censored{}, active{};
    bool mean_ci_available{};
    double mean_lines{}, median_lines{}, p05_lines{}, p95_lines{}, sd_lines{}, ci_low{}, ci_high{};
    double mean_pieces{}, median_pieces{}, p05_pieces{}, p95_pieces{};
};
Summary summarize(const std::vector<GameState>& games, std::size_t first, std::size_t count);
struct Survival { bool identifiable{}; double probability{}, ci_low{}, ci_high{}; };
Survival survival(const std::vector<GameState>& games, std::size_t first, std::size_t count, std::uint64_t milestone);
void validate_config(const Config& config);
} // namespace tetris
