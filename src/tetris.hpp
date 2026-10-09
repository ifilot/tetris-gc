#pragma once
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace tetris {
constexpr int width = 10, max_height = 22;
constexpr std::uint16_t full_row = (1U << width) - 1;
enum class Rules { article, battleblocks, article_drop };
// Costs in order: aggregate height, holes, bumpiness, completed lines.
// Lower is better; a useful line coefficient is negative.
using Weights = std::array<double, 4>;
Weights baseline(Rules rules);
Weights normalize(Weights weights);

// Explicit PRNG and bounded sampling make streams independent of thread scheduling.
class Random {
    std::uint64_t state_;
public:
    explicit Random(std::uint64_t seed) : state_(seed) {}
    std::uint64_t next();
    std::uint64_t bounded(std::uint64_t bound);
    double unit();
    std::uint64_t state() const { return state_; }
    void restore(std::uint64_t state) { state_ = state; }
};

struct Cell { int x{}, y{}; };
struct Shape {
    std::array<Cell, 4> cells{};
    std::array<int, 4> bottom{};
    int width{}, height{}, rotation{};
};
const std::array<std::vector<Shape>, 7>& shapes();
struct Features { int height{}, holes{}, bumpiness{}, lines{}; };
double score(const Features& features, const Weights& weights);
struct Placement {
    int rotation{-1}, x{}, y{};
    Features features{};
    double cost{};
    Shape shape{};
    bool valid() const { return rotation >= 0; }
};
struct Board {
    std::array<std::uint16_t, max_height> rows{}; // Bottom row first.
    std::array<int, width> heights{};
    int occupied{}, lines{}, limit{22};
    explicit Board(int height = 22) : limit(height) {}
    void rebuild();
    Features features(Rules rules) const;
    bool preview(const Shape& shape, int x, Rules rules, Placement& result) const;
    bool preview_at(const Shape& shape, int x, int y, Rules rules, Placement& result) const;
    void place(const Shape& shape, const Placement& placement);
    int clear_lines();
    bool topped_out(Rules rules) const;
};
Placement best_move(const Board& board, int piece, int next, const Weights& weights, Rules rules);
using Streams = std::vector<std::vector<std::uint8_t>>;
Streams make_streams(int games, int pieces, std::uint64_t seed);
struct GameResult { std::uint64_t lines{}, pieces{}; bool topped_out{}; };
struct Bag {
    Random random{0};
    std::array<std::uint8_t, 7> pieces{};
    std::uint8_t index{7};
    explicit Bag(std::uint64_t seed = 0) : random(seed) {}
    int next();
};
// status: 0 active, 1 topout, 2 stopped at the configured cap (censored).
struct GameState {
    Board board{};
    Bag bag{};
    std::uint64_t seed{}, lines{}, pieces{};
    int current{}, next{};
    std::uint8_t status{};
    GameState() = default;
    GameState(std::uint64_t seed, Rules rules);
    void advance(std::uint64_t target, const Weights& weights, Rules rules, const std::atomic<bool>* stop = nullptr);
};
std::vector<std::uint64_t> game_seeds(int games, std::uint64_t seed);
struct Evaluation;
std::vector<Evaluation> evaluate_seeded(const std::vector<Weights>& candidates,
    const std::vector<std::uint64_t>& seeds, std::uint64_t pieces, Rules rules,
    bool per_candidate_seeds = false, const std::atomic<bool>* stop = nullptr);
GameResult play(const std::vector<std::uint8_t>& stream, int max_pieces, const Weights& weights, Rules rules);
struct Evaluation {
    std::uint64_t lines{}, pieces{}, topouts{}, games{};
};
// Candidate x game parallelism; one writer per job and exact integer aggregation.
std::vector<Evaluation> evaluate(const std::vector<Weights>& candidates, const Streams& streams,
                                 int pieces, Rules rules);
struct Candidate { Weights weights{}; Evaluation evaluation{}; std::uint64_t id{}; };
void rank(std::vector<Candidate>& population);
std::vector<Candidate> reproduce(const std::vector<Candidate>& population, Random& random,
                                 std::uint64_t& next_id, double mutation_probability);
} // namespace tetris
