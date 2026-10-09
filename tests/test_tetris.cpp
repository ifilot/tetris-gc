#include "tetris.hpp"
#include <omp.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>

using namespace tetris;
static void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
using Grid = std::array<std::array<bool, width>, max_height>;
static Grid grid_of(const Board& b) {
    Grid g{};
    for (int y = 0; y < b.limit; ++y)
        for (int x = 0; x < width; ++x) g[y][x] = b.rows[y] & (1U << x);
    return g;
}
static Features slow_features(const Grid& g, int limit, Rules rules) {
    constexpr int penalties[] = {0,1,2,3,4,8,16,32,64,75,100,125};
    Features f{}; std::array<int, width> heights{};
    for (int x = 0; x < width; ++x) {
        bool above = false;
        for (int y = limit-1; y >= 0; --y) {
            if (g[y][x]) { if (!above) heights[x] = y+1; above = true; }
            else if (above) ++f.holes;
        }
        f.height += heights[x];
        if (x) { int diff = std::abs(heights[x]-heights[x-1]); f.bumpiness += rules != Rules::battleblocks ? diff : penalties[std::min(diff,11)]; }
    }
    for (int y = 0; y < limit; ++y) if (std::all_of(g[y].begin(), g[y].end(), [](bool v) { return v; })) ++f.lines;
    return f;
}
static bool equal(const Features& a, const Features& b) {
    return a.height == b.height && a.holes == b.holes && a.bumpiness == b.bumpiness && a.lines == b.lines;
}
static bool slow_place(Grid& g, int limit, const Shape& s, int x, int& landing) {
    auto valid = [&](int y) {
        for (auto c : s.cells) if (y+c.y < 0 || (y+c.y < limit && g[y+c.y][x+c.x])) return false;
        return true;
    };
    int y = limit;
    while (valid(y-1)) --y;
    if (y+s.height > limit) return false;
    for (auto c : s.cells) g[y+c.y][x+c.x] = true;
    landing = y; return true;
}
static Placement slow_best(const Board& board, int piece, int next, const Weights& w, Rules rules) {
    Placement best; best.cost = INFINITY;
    for (const auto& s : shapes()[piece]) for (int x = 0; x <= width-s.width; ++x) {
        auto g = grid_of(board); int y;
        if (!slow_place(g, board.limit, s, x, y)) continue;
        auto f = slow_features(g, board.limit, rules);
        double cost = score(f, w);
        if (rules == Rules::article_drop && next >= 0) {
            cost = INFINITY;
            for (const auto& ns : shapes()[next]) for (int nx = 0; nx <= width-ns.width; ++nx) {
                auto ng = g; int ny;
                if (slow_place(ng, board.limit, ns, nx, ny)) cost = std::min(cost, score(slow_features(ng,board.limit,rules), w));
            }
        }
        if (!best.valid() || cost < best.cost) best = {s.rotation,x,y,f,cost};
    }
    return best;
}
static void test_geometry() {
    const int counts[] = {2,1,4,2,2,4,4};
    for (int p = 0; p < 7; ++p) {
        check(shapes()[p].size() == static_cast<std::size_t>(counts[p]), "Unique rotation count");
        for (const auto& s : shapes()[p]) {
            unsigned mask = 0;
            for (auto c : s.cells) mask |= 1U << (c.y*4+c.x);
            check(std::popcount(mask) == 4, "Tetromino has four distinct cells");
            for (int x = 0; x <= width-s.width; ++x) {
                Board b; Placement result; check(b.preview(s,x,Rules::article,result), "Empty board placement");
                check(result.y == 0, "Empty board landing");
                b.place(s,result); check(b.occupied == 4, "Four occupied cells");
            }
        }
    }
}
static void test_reference() {
    Random rng(121);
    for (auto rules : {Rules::article_drop, Rules::battleblocks}) for (int trial = 0; trial < 120; ++trial) {
        Board b(rules != Rules::battleblocks ? 22 : 20);
        int height = static_cast<int>(rng.bounded(20));
        for (int y = 0; y < height; ++y) b.rows[y] = static_cast<std::uint16_t>(rng.bounded(1024));
        b.rebuild();
        check(equal(b.features(rules),slow_features(grid_of(b),b.limit,rules)), "Board features disagree with cell reference");
        for (int piece = 0; piece < 7; ++piece) {
            for (const auto& s : shapes()[piece]) for (int x = 0; x <= width-s.width; ++x) {
                Placement fast; bool ok = b.preview(s,x,rules,fast);
                auto g = grid_of(b); int y; bool slow_ok = slow_place(g,b.limit,s,x,y);
                check(ok == slow_ok, "Placement validity differs");
                if (ok) {
                    check(fast.y == y, "Landing differs");
                    check(equal(fast.features,slow_features(g,b.limit,rules)), "Placement features differ");
                    Board placed = b; placed.place(s,fast);
                    check(grid_of(placed) == g, "Placement bitboard differs");
                    check(equal(placed.features(rules),fast.features), "Cached features differ");
                    auto old_occupied = placed.occupied; auto removed = placed.clear_lines();
                    check(placed.occupied == old_occupied-width*removed && placed.lines == 0, "Line clear conservation");
                    check(equal(placed.features(rules),slow_features(grid_of(placed),placed.limit,rules)), "Post-clear features differ");
                }
            }
            if (trial < 12) {
                auto w = normalize(baseline(rules));
                auto fast = best_move(b,piece,(piece+2)%7,w,rules);
                auto slow = slow_best(b,piece,(piece+2)%7,w,rules);
                check(fast.rotation == slow.rotation && fast.x == slow.x && fast.y == slow.y && fast.cost == slow.cost,
                      "Lookahead move differs from cell reference");
            }
        }
    }
}
static void test_lines_and_topout() {
    Board b;
    b.rows[0] = full_row; b.rows[1] = full_row; b.rows[2] = 3; b.rows[4] = 4; b.rebuild();
    check(b.clear_lines() == 2 && b.rows[0] == 3 && b.rows[2] == 4 && b.occupied == 3, "Row compaction");
    b.rows[20] = 1; b.rebuild(); check(b.topped_out(Rules::article), "Hidden rows cause topout");
    b.rows[20] = full_row; b.rebuild(); b.clear_lines(); check(!b.topped_out(Rules::article), "Topout checked after clearing");
    Board well(20); for (int y = 0; y < 19; ++y) well.rows[y] = 1; well.rebuild();
    check(well.features(Rules::battleblocks).bumpiness == 125, "Deep well penalty saturates");
    auto stream = make_streams(1,500,8)[0];
    for (auto rules : {Rules::article, Rules::battleblocks}) {
        auto result = play(stream,500,normalize(baseline(rules)),rules);
        check(result.lines*10 <= result.pieces*4 && result.pieces <= 500 && result.pieces > 0, "Game piece conservation");
    }
}
static void test_streams_and_parallelism() {
    auto streams = make_streams(9,100,77);
    check(streams == make_streams(9,100,77), "Deterministic streams");
    check(streams != make_streams(9,100,78), "Different seed changes streams");
    for (auto stream : streams) for (std::size_t i = 0; i+7 <= stream.size(); i += 7) {
        unsigned mask = 0; for (int j = 0; j < 7; ++j) mask |= 1U << stream[i+j];
        check(mask == 127, "Seven-bag contents");
    }
    for (auto rules : {Rules::article,Rules::battleblocks}) {
        std::vector<Weights> weights{normalize(baseline(rules)), normalize({1,2,3,-4}), normalize({-1,1,-1,1})};
        omp_set_num_threads(1); auto a = evaluate(weights,streams,100,rules);
        omp_set_num_threads(4); auto b = evaluate(weights,streams,100,rules);
        for (std::size_t i = 0; i < a.size(); ++i)
            check(a[i].lines == b[i].lines && a[i].pieces == b[i].pieces && a[i].topouts == b[i].topouts, "Parallel reproducibility");
    }
}
static void test_genetics() {
    std::vector<Candidate> population;
    for (int i = 0; i < 10; ++i) population.push_back({normalize({double(i+1),2,3,-4}), {std::uint64_t(i),0,0,1},std::uint64_t(i)});
    rank(population); check(population.front().evaluation.lines == 9, "Fitness ordering");
    Random random(5); std::uint64_t next_id = 10;
    auto children = reproduce(population,random,next_id,1);
    check(children.size() == 3 && next_id == 13, "Thirty percent replacement");
    for (const auto& c : children) {
        double norm = 0; for (auto w : c.weights) norm += w*w;
        check(std::abs(norm-1) < 1e-12, "Children normalized");
    }
    for (auto& c : population) c.evaluation.lines = 0;
    children = reproduce(population,random,next_id,0);
    for (const auto& c : children) for (auto w : c.weights) check(std::isfinite(w), "Zero fitness crossover finite");
}
int main() {
    try {
        test_geometry(); test_reference(); test_lines_and_topout(); test_streams_and_parallelism(); test_genetics();
        std::cout << "All simulator, reference, genetic and concurrency checks passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
