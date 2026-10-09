#include "experiment.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
using namespace tetris;
void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
int main() {
    try {
        auto streams=make_streams(3,120,84);auto seeds=game_seeds(3,84);
        for(auto rules:{Rules::article,Rules::article_drop,Rules::battleblocks}) for(int i=0;i<3;++i) {
            GameState state(seeds[i],rules);auto w=normalize(baseline(rules));
            state.advance(35,w,rules);state.advance(120,w,rules);
            auto reference=play(streams[i],120,w,rules);
            check(state.lines==reference.lines && state.pieces==reference.pieces && (state.status==1)==reference.topped_out,
                  "Streaming generator differs from precomputed streams");
        }
        std::vector<GameState> data(4);
        data[0].pieces=10;data[0].status=1;data[0].lines=2;
        data[1].pieces=20;data[1].status=1;data[1].lines=5;
        data[2].pieces=30;data[2].status=2;data[2].lines=8;
        data[3].pieces=30;data[3].status=2;data[3].lines=9;
        auto s=summarize(data,0,4);
        check(s.mean_lines==6 && s.median_lines==6.5 && s.topouts==2 && s.censored==2 && s.mean_ci_available,"Summary moments");
        auto km=survival(data,0,4,20);
        check(km.identifiable && std::abs(km.probability-.5)<1e-12 && km.ci_low<.5 && km.ci_high>.5,"Survival at event horizon");
        check(!survival(data,0,4,31).identifiable,"No extrapolation beyond censoring");
        data[1].status=2;data[1].pieces=15;
        km=survival(data,0,4,20);
        check(std::abs(km.probability-.75)<1e-12,"Kaplan-Meier censoring");
        for(auto& g:data) { g.status=2;g.pieces=30; }
        km=survival(data,0,4,20);
        check(km.probability==1 && km.ci_low<1 && km.ci_high==1,"Zero deaths still have uncertainty");
        data[0].status=0;
        check(!summarize(data,0,4).mean_ci_available,"Incomplete run confidence flag");
        check(!summarize(data,0,1).mean_ci_available,"Single game confidence flag");
        std::cout << "Streaming and statistical checks passed\n";
    } catch(const std::exception& e) { std::cerr << e.what() << '\n';return 1; }
}
