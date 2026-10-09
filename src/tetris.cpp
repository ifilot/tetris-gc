#include "tetris.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>

namespace tetris {
Weights baseline(Rules rules) {
    return rules != Rules::battleblocks ? Weights{0.510066, 0.35663, 0.184483, -0.760666}
                                  : Weights{510, 869, 184, -761};
}
Weights normalize(Weights weights) {
    double norm = 0;
    for (double w : weights) norm = std::hypot(norm, w);
    if (!std::isfinite(norm) || norm == 0) throw std::invalid_argument("Weights must be finite and nonzero");
    for (double& w : weights) w /= norm;
    return weights;
}
std::uint64_t Random::next() {
    auto z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
std::uint64_t Random::bounded(std::uint64_t bound) {
    if (!bound) throw std::invalid_argument("Random bound must be positive");
    const auto threshold = -bound % bound;
    for (;;) { auto n = next(); if (n >= threshold) return n % bound; }
}
double Random::unit() { return static_cast<double>(next() >> 11) * 0x1.0p-53; }

const std::array<std::vector<Shape>, 7>& shapes() {
    static const auto table = [] {
        // I, O, T, S, Z, J, L in Battleblocks' initial orientation.
        std::array<std::array<Cell, 4>, 7> starts{{
            {{{0,0},{1,0},{2,0},{3,0}}}, {{{0,0},{1,0},{0,1},{1,1}}},
            {{{0,0},{1,0},{2,0},{1,1}}}, {{{0,0},{1,0},{1,1},{2,1}}},
            {{{0,1},{1,1},{1,0},{2,0}}}, {{{0,1},{0,0},{1,0},{2,0}}},
            {{{0,0},{1,0},{2,0},{2,1}}}
        }};
        std::array<std::vector<Shape>, 7> result;
        for (int p = 0; p < 7; ++p) {
            auto cells = starts[p];
            std::vector<std::uint16_t> seen;
            for (int r = 0; r < 4; ++r) {
                int minx = 100, miny = 100;
                for (auto c : cells) { minx = std::min(minx, c.x); miny = std::min(miny, c.y); }
                Shape s; s.rotation = r; s.bottom.fill(4);
                std::uint16_t mask = 0;
                for (int i = 0; i < 4; ++i) {
                    cells[i].x -= minx; cells[i].y -= miny;
                    auto c = cells[i]; s.cells[i] = c;
                    s.width = std::max(s.width, c.x + 1); s.height = std::max(s.height, c.y + 1);
                    s.bottom[c.x] = std::min(s.bottom[c.x], c.y);
                    mask |= 1U << (c.y * 4 + c.x);
                }
                if (std::find(seen.begin(), seen.end(), mask) == seen.end()) {
                    seen.push_back(mask); result[p].push_back(s);
                }
                for (auto& c : cells) { int oldx = c.x; c.x = c.y; c.y = -oldx; }
            }
        }
        return result;
    }();
    return table;
}
static int bump(const std::array<int, width>& heights, Rules rules) {
    constexpr int penalties[] = {0,1,2,3,4,8,16,32,64,75,100,125};
    int value = 0;
    for (int x = 1; x < width; ++x) {
        int d = std::abs(heights[x] - heights[x-1]);
        value += rules != Rules::battleblocks ? d : penalties[std::min(d, 11)];
    }
    return value;
}
double score(const Features& f, const Weights& w) {
    return -(-w[0]*f.height - w[3]*f.lines - w[1]*f.holes - w[2]*f.bumpiness);
}
void Board::rebuild() {
    heights.fill(0); occupied = 0; lines = 0;
    for (int y = 0; y < limit; ++y) {
        occupied += std::popcount(rows[y]); lines += rows[y] == full_row;
        unsigned mask = rows[y];
        while (mask) { int x = std::countr_zero(mask); heights[x] = y + 1; mask &= mask - 1; }
    }
}
Features Board::features(Rules rules) const {
    int h = std::accumulate(heights.begin(), heights.end(), 0);
    return {h, h - occupied, bump(heights, rules), lines};
}
bool Board::preview(const Shape& s, int x, Rules rules, Placement& result) const {
    if (x < 0 || x + s.width > width) return false;
    int y = 0;
    for (int col = 0; col < s.width; ++col) y = std::max(y, heights[x+col] - s.bottom[col]);
    return preview_at(s, x, y, rules, result);
}
bool Board::preview_at(const Shape& s, int x, int y, Rules rules, Placement& result) const {
    if (x < 0 || x+s.width > width || y < 0 || y+s.height > limit) return false;
    auto h = heights;
    std::array<std::uint16_t, 4> touched{};
    for (auto c : s.cells) {
        if (rows[y+c.y] & (1U << (x+c.x))) return false;
        h[x+c.x] = std::max(h[x+c.x], y+c.y+1);
        touched[c.y] |= 1U << (x+c.x);
    }
    int completed = lines;
    for (int row = 0; row < s.height; ++row)
        if (touched[row] && rows[y+row] != full_row && (rows[y+row] | touched[row]) == full_row) ++completed;
    int total = std::accumulate(h.begin(), h.end(), 0);
    result = {s.rotation, x, y, {total, total-occupied-4, bump(h, rules), completed}, 0};
    result.shape = s;
    return true;
}
void Board::place(const Shape& s, const Placement& p) {
    for (auto c : s.cells) {
        int x = p.x+c.x, y = p.y+c.y;
        rows[y] |= 1U << x; heights[x] = std::max(heights[x], y+1);
    }
    occupied += 4; lines = p.features.lines;
}
int Board::clear_lines() {
    if (!lines) return 0;
    int write = 0, cleared = 0;
    for (int y = 0; y < limit; ++y) {
        if (rows[y] == full_row) ++cleared;
        else rows[write++] = rows[y];
    }
    std::fill(rows.begin()+write, rows.end(), 0);
    rebuild();
    return cleared;
}
bool Board::topped_out(Rules rules) const {
    return rules != Rules::battleblocks && (rows[20] || rows[21]);
}
namespace {
// Square matrices and kick order match the browser source, whose kicks are
// simpler than the standard SRS tables advertised in the article.
struct SpawnPiece {
    std::array<Cell,4> cells{}; // Coordinates measured downward from matrix top.
    int dimension{}, row{}, column{}, rotation{};
};
SpawnPiece spawn(int piece) {
    SpawnPiece p;
    p.dimension = piece == 0 ? 4 : piece == 1 ? 2 : 3;
    p.column = (width-p.dimension)/2;
    const auto& s = shapes()[piece][0];
    for (int i = 0; i < 4; ++i) p.cells[i] = {s.cells[i].x, piece == 0 ? 1 : s.height-1-s.cells[i].y};
    return p;
}
bool valid_spawn(const Board& b, const SpawnPiece& p) {
    for (auto c : p.cells) {
        int x = p.column+c.x, y = b.limit-1-p.row-c.y;
        if (x < 0 || x >= width || y < 0 || y >= b.limit || (b.rows[y] & (1U << x))) return false;
    }
    return true;
}
void rotate_spawn(const Board& b, SpawnPiece& p) {
    auto rotated = p;
    for (auto& c : rotated.cells) { auto x = c.x; c.x = p.dimension-1-c.y; c.y = x; }
    rotated.rotation = (p.rotation+1)%4;
    if (valid_spawn(b,rotated)) { p = rotated; return; }
    for (int sign : {1,-1}) for (int dx = 0; dx < p.dimension-1; ++dx) {
        rotated.column = p.column+sign*dx; rotated.row = p.row;
        if (valid_spawn(b,rotated)) { p = rotated; return; }
        for (int dy = 0; dy < p.dimension-1; ++dy) {
            rotated.row = p.row-dy;
            if (valid_spawn(b,rotated)) { p = rotated; return; }
        }
    }
}
template<class Callback> void article_placements(const Board& b, int piece, Callback callback) {
    // At low stack heights every spawn is unobstructed; retain the fast skyline kernel.
    if (!(b.rows[18] | b.rows[19] | b.rows[20] | b.rows[21])) {
        for (const auto& s : shapes()[piece]) for (int x = 0; x <= width-s.width; ++x) {
            Placement p; if (b.preview(s,x,Rules::article,p)) callback(p);
        }
        return;
    }
    for (int rotation = 0; rotation < 4; ++rotation) {
        auto p = spawn(piece);
        for (int i = 0; i < rotation; ++i) rotate_spawn(b,p);
        auto left = p; --left.column;
        while (valid_spawn(b,left)) { p = left; --left.column; }
        while (valid_spawn(b,p)) {
            auto landed = p, down = p; ++down.row;
            while (valid_spawn(b,down)) { landed = down; ++down.row; }
            int minx = 4, minrow = 4, maxrow = 0;
            for (auto c : landed.cells) { minx = std::min(minx,c.x); minrow = std::min(minrow,c.y); maxrow = std::max(maxrow,c.y); }
            Shape s; s.rotation = p.rotation; s.height = maxrow-minrow+1; s.bottom.fill(4);
            for (int i = 0; i < 4; ++i) {
                auto c = landed.cells[i]; c = {c.x-minx,maxrow-c.y}; s.cells[i] = c;
                s.width = std::max(s.width,c.x+1); s.bottom[c.x] = std::min(s.bottom[c.x],c.y);
            }
            Placement result;
            if (b.preview_at(s,landed.column+minx,b.limit-1-landed.row-maxrow,Rules::article,result)) callback(result);
            ++p.column;
        }
    }
}
struct ArticleChoice { Placement placement; std::optional<double> cost; };
ArticleChoice article_best(const Board& b, int piece, int next, const Weights& weights) {
    ArticleChoice best;
    article_placements(b,piece,[&](const Placement& p) {
        std::optional<double> cost;
        if (next >= 0) {
            auto after = b; after.place(p.shape,p);
            cost = article_best(after,next,-1,weights).cost;
        } else cost = score(p.features,weights);
        // The upstream recursive scorer returns null when a next piece cannot
        // spawn. JavaScript compares null as zero and resets a null best score.
        if (!best.cost || cost.value_or(0) < *best.cost) best = {p,cost};
    });
    best.placement.cost = best.cost.value_or(0);
    return best;
}
} // namespace
Placement best_move(const Board& board, int piece, int next, const Weights& weights, Rules rules) {
    if (rules == Rules::article) return article_best(board,piece,next,weights).placement;
    Placement best; best.cost = std::numeric_limits<double>::infinity();
    for (const auto& s : shapes()[piece]) {
        for (int x = 0; x <= width-s.width; ++x) {
            Placement p;
            if (!board.preview(s, x, rules, p)) continue;
            double cost = score(p.features, weights);
            if (rules == Rules::article_drop && next >= 0) {
                Board after = board; after.place(s, p);
                // Upstream AI evaluates both pieces before clearing any lines.
                cost = std::numeric_limits<double>::infinity();
                for (const auto& ns : shapes()[next]) {
                    for (int nx = 0; nx <= width-ns.width; ++nx) {
                        Placement np;
                        if (after.preview(ns, nx, rules, np)) cost = std::min(cost, score(np.features, weights));
                    }
                }
            }
            if (!best.valid() || cost < best.cost) { best = p; best.cost = cost; }
        }
    }
    return best;
}
Streams make_streams(int games, int pieces, std::uint64_t seed) {
    Streams streams(games);
    Random seeds(seed);
    for (auto& stream : streams) {
        Random random(seeds.next()); stream.reserve(static_cast<std::size_t>(pieces)+1);
        while (stream.size() < static_cast<std::size_t>(pieces)+1) {
            std::array<std::uint8_t, 7> bag{0,1,2,3,4,5,6};
            for (int i = 6; i > 0; --i) std::swap(bag[i], bag[random.bounded(i+1)]);
            for (auto p : bag) if (stream.size() < static_cast<std::size_t>(pieces)+1) stream.push_back(p);
        }
    }
    return streams;
}
GameResult play(const std::vector<std::uint8_t>& stream, int max_pieces, const Weights& weights, Rules rules) {
    Board board(rules != Rules::battleblocks ? 22 : 20);
    GameResult result;
    for (int i = 0; i < max_pieces; ++i) {
        const int piece = stream[i];
        auto p = best_move(board, piece, stream[i+1], weights, rules);
        if (!p.valid()) { result.topped_out = true; break; }
        board.place(p.shape, p); ++result.pieces;
        result.lines += board.clear_lines();
        if (board.topped_out(rules)) { result.topped_out = true; break; }
    }
    return result;
}
int Bag::next() {
    if (index == 7) {
        pieces = {0,1,2,3,4,5,6};
        for (int i = 6; i > 0; --i) std::swap(pieces[i],pieces[random.bounded(i+1)]);
        index = 0;
    }
    return pieces[index++];
}
std::vector<std::uint64_t> game_seeds(int games, std::uint64_t seed) {
    Random random(seed); std::vector<std::uint64_t> seeds(games);
    for (auto& s : seeds) s = random.next();
    return seeds;
}
GameState::GameState(std::uint64_t initial_seed, Rules rules)
    : board(rules == Rules::battleblocks ? 20 : 22), bag(initial_seed), seed(initial_seed) {
    current = bag.next(); next = bag.next();
}
void GameState::advance(std::uint64_t target, const Weights& weights, Rules rules, const std::atomic<bool>* stop) {
    while (!status && pieces < target) {
        if (stop && !(pieces & 1023) && stop->load(std::memory_order_relaxed)) break;
        auto p = best_move(board,current,next,weights,rules);
        if (!p.valid()) { status = 1; break; }
        board.place(p.shape,p); ++pieces; lines += board.clear_lines();
        current = next; next = bag.next();
        if (board.topped_out(rules)) status = 1;
    }
}
std::vector<Evaluation> evaluate_seeded(const std::vector<Weights>& candidates,
    const std::vector<std::uint64_t>& seeds, std::uint64_t pieces, Rules rules,
    bool per_candidate_seeds, const std::atomic<bool>* stop) {
    (void)shapes();
    const auto games = static_cast<std::int64_t>(per_candidate_seeds ? seeds.size()/candidates.size() : seeds.size());
    const auto jobs = static_cast<std::int64_t>(candidates.size())*games;
    std::vector<GameResult> results(jobs);
    #pragma omp parallel for schedule(dynamic, 8)
    for (std::int64_t job = 0; job < jobs; ++job) {
        if (stop && stop->load(std::memory_order_relaxed)) continue;
        GameState state(seeds[per_candidate_seeds ? job : job%games],rules);
        state.advance(pieces,candidates[job/games],rules,stop);
        results[job] = {state.lines,state.pieces,state.status == 1};
    }
    std::vector<Evaluation> totals(candidates.size());
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        totals[c].games = games;
        for (std::int64_t g = 0; g < games; ++g) {
            auto r = results[c*games+g]; totals[c].lines += r.lines;
            totals[c].pieces += r.pieces; totals[c].topouts += r.topped_out;
        }
    }
    return totals;
}
std::vector<Evaluation> evaluate(const std::vector<Weights>& candidates, const Streams& streams,
                                 int pieces, Rules rules) {
    // Initialize static geometry before entering the parallel region.
    (void)shapes();
    const auto games = static_cast<std::int64_t>(streams.size());
    const auto jobs = static_cast<std::int64_t>(candidates.size()) * games;
    std::vector<GameResult> results(static_cast<std::size_t>(jobs));
    #pragma omp parallel for schedule(dynamic, 8)
    for (std::int64_t job = 0; job < jobs; ++job)
        results[job] = play(streams[job % games], pieces, candidates[job / games], rules);
    std::vector<Evaluation> totals(candidates.size());
    for (std::size_t c = 0; c < candidates.size(); ++c) {
        auto& total = totals[c]; total.games = games;
        for (std::int64_t g = 0; g < games; ++g) {
            const auto& r = results[c*games+g];
            total.lines += r.lines; total.pieces += r.pieces; total.topouts += r.topped_out;
        }
    }
    return totals;
}
void rank(std::vector<Candidate>& population) {
    std::sort(population.begin(), population.end(), [](const Candidate& a, const Candidate& b) {
        return a.evaluation.lines != b.evaluation.lines ? a.evaluation.lines > b.evaluation.lines : a.id < b.id;
    });
}
std::vector<Candidate> reproduce(const std::vector<Candidate>& population, Random& random,
                                 std::uint64_t& next_id, double mutation_probability) {
    const int count = static_cast<int>(population.size());
    const int offspring = std::max(1, count*3/10), ways = std::max(2, count/10);
    std::vector<Candidate> children; children.reserve(offspring);
    std::vector<int> pool(count);
    for (int child = 0; child < offspring; ++child) {
        std::iota(pool.begin(), pool.end(), 0);
        int first = count, second = count;
        // Partial Fisher-Yates: tournament sampling without replacement.
        for (int i = 0; i < ways; ++i) {
            int pick = i + static_cast<int>(random.bounded(count-i)); std::swap(pool[i], pool[pick]);
            int idx = pool[i];
            if (idx < first) { second = first; first = idx; }
            else if (idx < second) second = idx;
        }
        const auto& a = population[first]; const auto& b = population[second];
        double fa = static_cast<double>(a.evaluation.lines), fb = static_cast<double>(b.evaluation.lines);
        if (fa + fb == 0) fa = fb = 1;
        Weights weights{};
        for (int k = 0; k < 4; ++k) weights[k] = (fa*a.weights[k] + fb*b.weights[k]) / (fa+fb);
        double norm = 0; for (auto w : weights) norm += w*w;
        weights = norm < 1e-24 ? a.weights : normalize(weights);
        if (random.unit() < mutation_probability) weights[random.bounded(4)] += random.unit()*0.4-0.2;
        children.push_back({normalize(weights), {}, next_id++});
    }
    return children;
}
} // namespace tetris
