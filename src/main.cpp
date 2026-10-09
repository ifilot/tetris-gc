#include "experiment.hpp"
#include <omp.h>
#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <string_view>

namespace {
using namespace tetris;
using Clock=std::chrono::steady_clock;
std::atomic<bool> interrupted{false};
static_assert(std::atomic<bool>::is_always_lock_free);
void interrupt_handler(int) { interrupted.store(true,std::memory_order_relaxed); }
struct Options {
    Config config;
    std::string command{"train"},output,input,resume,weights_file;
    int threads{};
    std::optional<Weights> weights;
    std::set<std::string> specified;
    bool verbose{},compare{true};
};
void help() {
    std::cout << R"(tetris-gc: OpenMP genetic optimization and endurance experiments
Usage: tetris-gc [train|evaluate|endurance|benchmark|report] [options]
  --rules article|battleblocks|article-drop  article matches the browser source
  --population N          Training population (1000)
  --games N               Games per candidate/policy (100)
  --pieces N              64-bit piece cap (500; endurance defaults to 10000000)
  --generations N         Replacement generations (20)
  --runs N                Independent training seeds seed, seed+1, ... (1)
  --seed N                Experiment seed (1)
  --sampling common|independent|refresh   Training game sampling (common)
  --mutation P            Mutation probability (0.05)
  --validation-games N    Paired winner/baseline validation games (100; 0 skips)
  --validation-pieces N   Validation piece cap (500)
  --checkpoint-pieces N   Pieces per game between endurance snapshots (100000)
  --threads N             OpenMP workers (0 respects runtime settings)
  --weights H,O,B,L       Explicit costs for evaluation/endurance
  --weights-file PATH     Binary training checkpoint; evaluate every run's winner
  --no-baseline           Skip the paired published baseline
  --output PATH           Binary checkpoint/results (.tgc)
  --resume PATH           Resume training or unfinished endurance states
  --input PATH            Input file for report
  --verbose               Print generation progress
  --help                  Show help

Defaults: results/train.tgc, results/evaluation.tgc, results/endurance.tgc.
report writes results/statistics.tgc and prints a brief summary.
Ctrl+C saves resumable state; output is little-endian binary with a checksum.
)",std::cout.flush();
}
template<class T> T number(std::string_view text) {
    T value{};auto [end,ec]=std::from_chars(text.data(),text.data()+text.size(),value);
    if(ec!=std::errc{} || end!=text.data()+text.size()) throw std::invalid_argument("Invalid number: "+std::string(text));
    return value;
}
Weights parse_weights(std::string_view text) {
    Weights weights{};
    for(int i=0;i<4;++i) { auto comma=text.find(',');
        if((i<3 && comma==text.npos)||(i==3 && comma!=text.npos)) throw std::invalid_argument("Expected four weights: H,O,B,L");
        weights[i]=number<double>(text.substr(0,comma));if(i<3) text.remove_prefix(comma+1);
    }return normalize(weights);
}
Options parse(int argc,char** argv) {
    Options o;int i=1;if(i<argc && argv[i][0]!='-') o.command=argv[i++];
    const std::set<std::string> commands{"train","evaluate","endurance","benchmark","report"};
    if(!commands.contains(o.command)) throw std::invalid_argument("Unknown command: "+o.command);
    for(;i<argc;++i) {
        std::string key=argv[i];o.specified.insert(key);
        if(key=="--help" || key=="-h") { help();std::exit(0); }
        if(key=="--verbose") { o.verbose=true;continue; }
        if(key=="--no-baseline") { o.compare=false;continue; }
        if(++i==argc) throw std::invalid_argument("Missing value for "+key);
        std::string value=argv[i];
        auto& c=o.config;
        if(key=="--population") c.population=number<std::uint32_t>(value);
        else if(key=="--games") c.games=number<std::uint32_t>(value);
        else if(key=="--pieces") c.pieces=number<std::uint64_t>(value);
        else if(key=="--generations") c.generations=number<std::uint32_t>(value);
        else if(key=="--runs") c.runs=number<std::uint32_t>(value);
        else if(key=="--validation-games") c.validation_games=number<std::uint32_t>(value);
        else if(key=="--validation-pieces") c.validation_pieces=number<std::uint64_t>(value);
        else if(key=="--checkpoint-pieces") c.checkpoint_pieces=number<std::uint64_t>(value);
        else if(key=="--seed") c.seed=number<std::uint64_t>(value);
        else if(key=="--mutation") c.mutation=number<double>(value);
        else if(key=="--threads") o.threads=number<int>(value);
        else if(key=="--output") o.output=value;
        else if(key=="--input") o.input=value;
        else if(key=="--resume") o.resume=value;
        else if(key=="--weights") o.weights=parse_weights(value);
        else if(key=="--weights-file") o.weights_file=value;
        else if(key=="--sampling") {
            if(value=="common") c.sampling=0;else if(value=="independent") c.sampling=1;else if(value=="refresh") c.sampling=2;
            else throw std::invalid_argument("Unknown sampling mode");
        } else if(key=="--rules") {
            if(value=="article") c.rules=Rules::article;else if(value=="battleblocks") c.rules=Rules::battleblocks;
            else if(value=="article-drop") c.rules=Rules::article_drop;else throw std::invalid_argument("Unknown rules");
        } else throw std::invalid_argument("Unknown option: "+key);
    }
    if(o.command=="endurance" && !o.specified.contains("--pieces")) o.config.pieces=10000000;
    validate_config(o.config);
    if(o.threads<0 || (o.weights && !o.weights_file.empty())) throw std::invalid_argument("Invalid threads or conflicting weights inputs");
    if(o.command=="train" && (o.weights || !o.weights_file.empty())) throw std::invalid_argument("Training starts with a random population");
    if(o.command=="report" && o.input.empty()) throw std::invalid_argument("report requires --input");
    if(!o.input.empty() && o.command!="report") throw std::invalid_argument("--input is only for report");
    if(!o.resume.empty() && (o.command=="report" || o.command=="benchmark")) throw std::invalid_argument("This command cannot resume");
    if(o.output.empty()) o.output=!o.resume.empty()?o.resume:"results/"+(o.command=="train"?std::string("train"):o.command=="report"?std::string("statistics"):o.command=="endurance"?std::string("endurance"):std::string("evaluation"))+".tgc";
    for(const auto& source:{o.input,o.weights_file}) if(!source.empty() && std::filesystem::absolute(source).lexically_normal()==std::filesystem::absolute(o.output).lexically_normal())
        throw std::invalid_argument("Output would overwrite the input file");
    return o;
}
Experiment resume(const Options& o,FileKind kind) {
    auto e=load_binary(o.resume);if(e.kind!=kind) throw std::invalid_argument("Wrong checkpoint kind for this command");
    for(const auto& key:o.specified) {
        bool allowed=key=="--resume" || key=="--output" || key=="--threads" || key=="--verbose" || key=="--checkpoint-pieces" ||
            (kind==FileKind::training?key=="--generations":key=="--pieces");
        if(!allowed) throw std::invalid_argument("Resume restores configuration; cannot override "+key);
    }
    if(o.specified.contains("--checkpoint-pieces")) e.config.checkpoint_pieces=o.config.checkpoint_pieces;
    if(kind==FileKind::training && o.specified.contains("--generations")) {
        if(o.config.generations<e.config.generations) throw std::invalid_argument("Generation target can only increase");
        if(o.config.generations>e.config.generations) for(auto& r:e.training) r.validation.clear();
        e.config.generations=o.config.generations;
    }
    if(kind==FileKind::endurance && o.specified.contains("--pieces")) {
        if(o.config.pieces<e.config.pieces) throw std::invalid_argument("Piece cap can only increase");
        e.config.pieces=o.config.pieces;
        for(auto& g:e.games) if(g.status==2 && g.pieces<e.config.pieces) g.status=0;
    }
    validate_config(e.config);return e;
}
void initialize_games(std::vector<GameState>& states,const std::vector<Weights>& policies,int games,std::uint64_t seed,Rules rules) {
    const auto seeds=game_seeds(games,seed);states.reserve(policies.size()*games);
    for(std::size_t p=0;p<policies.size();++p) for(auto s:seeds) states.emplace_back(s,rules);
}
void advance_games(std::vector<GameState>& states,const std::vector<Weights>& weights,std::uint32_t games,
                   std::uint64_t cap,std::uint64_t chunk,Rules rules) {
    #pragma omp parallel for schedule(dynamic, 1)
    for(std::int64_t i=0;i<static_cast<std::int64_t>(states.size());++i) {
        if(interrupted.load(std::memory_order_relaxed)) continue;
        auto& g=states[i];
        auto target=g.pieces+std::min(chunk,cap-g.pieces);
        g.advance(target,weights[i/games],rules,&interrupted);
        if(!g.status && g.pieces==cap) g.status=2;
    }
}
bool complete(const std::vector<GameState>& states) {
    return std::none_of(states.begin(),states.end(),[](const auto& g) { return g.status==0; });
}
std::uint64_t training_stream_seed(const TrainingRun& run,const Config& c,std::uint32_t generation) {
    if(c.sampling!=2) return run.seed;
    Random random(run.seed ^ (0x9e3779b97f4a7c15ULL*generation));return random.next();
}
void train(const Options& o) {
    Experiment e=o.resume.empty()?Experiment{}:resume(o,FileKind::training);
    if(o.resume.empty()) e.config=o.config;
    const auto& c=e.config;
    for(std::uint32_t index=0;index<c.runs && !interrupted.load();++index) {
        if(index==e.training.size()) {
            TrainingRun run;run.seed=c.seed+index;Random random(run.seed ^ 0xa0761d6478bd642fULL);
            for(std::uint32_t i=0;i<c.population;++i) {
                Weights w{};do { for(auto& v:w) v=random.unit()-.5; } while(std::all_of(w.begin(),w.end(),[](double v){return v==0;}));
                run.population.push_back({normalize(w),{},run.next_id++});
            }
            run.random_state=random.state();e.training.push_back(std::move(run));save_binary(o.output,e);
        }
        auto& run=e.training[index];
        for(std::uint32_t generation=run.generation+1;generation<=c.generations && !interrupted.load();++generation) {
            Random random(run.random_state);auto next_id=run.next_id;
            auto candidates=generation?reproduce(run.population,random,next_id,c.mutation):run.population;
            std::vector<Candidate> survivors;
            if(generation) { survivors=run.population;survivors.resize(survivors.size()-candidates.size()); }
            // Refreshed streams require re-evaluating survivors, so comparisons use the same batch.
            if(c.sampling==2) { candidates.insert(candidates.end(),survivors.begin(),survivors.end());survivors.clear(); }
            std::vector<Weights> weights;for(const auto& candidate:candidates) weights.push_back(candidate.weights);
            auto seeds=game_seeds(c.games,training_stream_seed(run,c,generation));
            if(c.sampling==1) {
                seeds.clear();seeds.reserve(candidates.size()*c.games);
                for(const auto& candidate:candidates) {
                    Random stream_seed(run.seed ^ (candidate.id*0xd1b54a32d192ed03ULL));
                    auto stream=game_seeds(c.games,stream_seed.next());seeds.insert(seeds.end(),stream.begin(),stream.end());
                }
            }
            auto evaluations=evaluate_seeded(weights,seeds,c.pieces,c.rules,c.sampling==1,&interrupted);
            if(interrupted.load()) break; // Current generation is discarded; RNG state is not committed.
            std::uint64_t evaluated_pieces=0;
            for(std::size_t i=0;i<candidates.size();++i) {
                candidates[i].evaluation=evaluations[i];
                if(evaluations[i].pieces>std::numeric_limits<std::uint64_t>::max()-evaluated_pieces)
                    throw std::overflow_error("Generation piece counter overflow");
                evaluated_pieces+=evaluations[i].pieces;
            }
            survivors.insert(survivors.end(),candidates.begin(),candidates.end());rank(survivors);run.population=std::move(survivors);
            run.random_state=random.state();run.next_id=next_id;run.generation=generation;run.validation.clear();
            double mean=0;for(const auto& candidate:run.population) mean+=static_cast<double>(candidate.evaluation.lines)/c.population;
            run.history.push_back({generation,run.population.front().evaluation.lines,evaluated_pieces,mean,run.population.front().weights});
            save_binary(o.output,e);
            if(o.verbose) std::cout << "run=" << index << " generation=" << generation << " best=" << run.history.back().best << '\n' << std::flush;
        }
        if(interrupted.load()) break;
        if(c.validation_games) {
            const std::vector<Weights> weights{run.population.front().weights,normalize(baseline(c.rules))};
            if(run.validation.empty()) initialize_games(run.validation,weights,c.validation_games,run.seed ^ 0xd1b54a32d192ed03ULL,c.rules);
            while(!complete(run.validation) && !interrupted.load()) {
                advance_games(run.validation,weights,c.validation_games,c.validation_pieces,c.checkpoint_pieces,c.rules);save_binary(o.output,e);
            }
        }
    }
    save_binary(o.output,e);
    std::cout << (interrupted.load()?"Paused: ":"Saved: ") << o.output << " (" << e.training.size() << " training runs)\n";
}
Experiment make_evaluation(const Options& o) {
    Experiment e;e.kind=FileKind::endurance;e.config=o.config;
    if(!o.weights_file.empty()) {
        auto trained=load_binary(o.weights_file);
        if(trained.kind!=FileKind::training) throw std::invalid_argument("--weights-file requires a training checkpoint");
        if(o.specified.contains("--rules") && e.config.rules!=trained.config.rules) throw std::invalid_argument("Training and evaluation rules differ");
        e.config.rules=trained.config.rules;
        for(const auto& r:trained.training) if(r.generation>=0) e.policies.push_back({r.seed,r.population.front().weights});
        if(e.policies.empty()) throw std::invalid_argument("Checkpoint has no evaluated winner");
    } else if(o.weights) e.policies.push_back({o.config.seed,*o.weights});
    if(e.policies.empty() || o.compare) e.policies.push_back({0,normalize(baseline(e.config.rules)),true});
    std::vector<Weights> weights;for(auto p:e.policies) weights.push_back(p.weights);
    initialize_games(e.games,weights,e.config.games,e.config.seed,e.config.rules);return e;
}
void evaluate(const Options& o) {
    const auto start=Clock::now();
    auto e=o.resume.empty()?make_evaluation(o):resume(o,FileKind::endurance);
    std::vector<Weights> weights;for(auto p:e.policies) weights.push_back(p.weights);
    if(o.command!="benchmark") save_binary(o.output,e);
    while(!complete(e.games) && !interrupted.load()) {
        advance_games(e.games,weights,e.config.games,e.config.pieces,e.config.checkpoint_pieces,e.config.rules);
        if(o.command!="benchmark") save_binary(o.output,e);
    }
    const double elapsed=std::chrono::duration<double>(Clock::now()-start).count();
    if(o.command=="benchmark") {
        std::uint64_t pieces=0;for(const auto& g:e.games) pieces+=g.pieces;
        std::cout << "pieces/s=" << pieces/std::max(elapsed,1e-9) << " elapsed=" << elapsed << "s\n";
    } else std::cout << (interrupted.load()?"Paused: ":"Saved: ") << o.output << " (" << e.games.size() << " game records)\n";
}
void report(const Options& o) {
    auto source=load_binary(o.input);Experiment e;
    if(source.kind==FileKind::training) {
        e.kind=FileKind::statistics;e.config=source.config;e.config.games=source.config.validation_games;e.config.pieces=source.config.validation_pieces;
        if(!e.config.games) throw std::invalid_argument("Training validation was disabled");
        for(const auto& r:source.training) if(!r.validation.empty()) {
            e.policies.push_back({r.seed,r.population.front().weights});e.policies.push_back({0,normalize(baseline(e.config.rules)),true});
            e.games.insert(e.games.end(),r.validation.begin(),r.validation.end());
        }
        if(e.policies.empty()) throw std::invalid_argument("No validation records yet");
    } else { e=std::move(source);e.kind=FileKind::statistics; }
    for(std::size_t p=0;p<e.policies.size();++p) {
        auto s=summarize(e.games,p*e.config.games,e.config.games);
        std::cout << "policy=" << p << (e.policies[p].baseline?" baseline":" winner") << " mean-lines=" << s.mean_lines << " median=" << s.median_lines
                  << " topouts=" << s.topouts << " capped=" << s.censored << " active=" << s.active << '\n';
    }
    save_binary(o.output,e);
    std::cout << "Saved: " << o.output << '\n';
}
} // namespace
int main(int argc,char** argv) {
    try {
        auto o=parse(argc,argv);omp_set_dynamic(0);if(o.threads) omp_set_num_threads(o.threads);
        std::signal(SIGINT,interrupt_handler);std::signal(SIGTERM,interrupt_handler);std::cout << std::setprecision(8);
        if(o.command=="train") train(o);else if(o.command=="report") report(o);else evaluate(o);
        return interrupted.load()?130:0;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n';return 1; }
}
