#include "experiment.hpp"
#include <algorithm>
#include <bit>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace tetris {
namespace {
using Bytes = std::vector<std::uint8_t>;
constexpr std::uint32_t format_version = 2;
constexpr std::uint64_t maximum_payload = 1ULL << 30;
static_assert(sizeof(double)==8 && std::numeric_limits<double>::is_iec559);
struct Writer {
    Bytes data;
    void u64(std::uint64_t v) { for (int i=0;i<8;++i) { data.push_back(v & 255); v >>= 8; } }
    void u32(std::uint32_t v) { for (int i=0;i<4;++i) { data.push_back(v & 255); v >>= 8; } }
    void f64(double v) { u64(std::bit_cast<std::uint64_t>(v)); }
    void weights(const Weights& w) { for (auto v : w) f64(v); }
};
struct Reader {
    const Bytes& data; std::size_t pos{};
    void need(std::size_t n) { if (n > data.size()-pos) throw std::runtime_error("Truncated binary file"); }
    std::uint64_t u64() { need(8); std::uint64_t v=0; for(int i=0;i<8;++i) v |= std::uint64_t(data[pos++])<<(8*i); return v; }
    std::uint32_t u32() { need(4); std::uint32_t v=0; for(int i=0;i<4;++i) v |= std::uint32_t(data[pos++])<<(8*i); return v; }
    double f64() { double v=std::bit_cast<double>(u64()); if(!std::isfinite(v)) throw std::runtime_error("Nonfinite binary value"); return v; }
    std::size_t count(std::size_t minimum_record) { auto n=u64(); if(n > (data.size()-pos)/minimum_record) throw std::runtime_error("Invalid binary record count"); return n; }
    Weights weights() { Weights w{}; for(auto& v:w) v=f64(); double norm=0; for(auto v:w) norm+=v*v; if(std::abs(norm-1)>1e-10) throw std::runtime_error("Invalid binary weights"); return w; }
};
std::uint64_t checksum(const Bytes& data) {
    std::uint64_t h=14695981039346656037ULL;
    for(auto b:data) { h^=b; h*=1099511628211ULL; } return h;
}
void put_config(Writer& w,const Config& c) {
    w.u32(static_cast<std::uint32_t>(c.rules)); w.u64(c.seed); w.u64(c.pieces);
    w.u64(c.validation_pieces); w.u64(c.checkpoint_pieces);
    w.u32(c.population); w.u32(c.games); w.u32(c.generations); w.u32(c.runs); w.u32(c.validation_games);
    w.f64(c.mutation); w.u32(c.sampling);
}
Config get_config(Reader& r) {
    Config c; c.rules=static_cast<Rules>(r.u32()); c.seed=r.u64(); c.pieces=r.u64();
    c.validation_pieces=r.u64(); c.checkpoint_pieces=r.u64();
    c.population=r.u32(); c.games=r.u32(); c.generations=r.u32(); c.runs=r.u32(); c.validation_games=r.u32();
    c.mutation=r.f64(); c.sampling=r.u32(); validate_config(c); return c;
}
void put_evaluation(Writer& w,const Evaluation& e) { w.u64(e.lines);w.u64(e.pieces);w.u64(e.topouts);w.u64(e.games); }
Evaluation get_evaluation(Reader& r) { Evaluation e; e.lines=r.u64();e.pieces=r.u64();e.topouts=r.u64();e.games=r.u64();return e; }
void put_game(Writer& w,const GameState& g) {
    w.u64(g.seed); w.u64(g.lines); w.u64(g.pieces); w.u64(g.bag.random.state());
    w.u32(g.status); w.u32(g.current); w.u32(g.next); w.u32(g.bag.index);
    for(auto p:g.bag.pieces) w.u32(p);
    for(auto row:g.board.rows) w.u32(row);
}
GameState get_game(Reader& r,Rules rules) {
    GameState g; g.seed=r.u64();g.lines=r.u64();g.pieces=r.u64();g.bag.random.restore(r.u64());
    auto status=r.u32(); auto current=r.u32();auto next=r.u32();auto index=r.u32();
    if(status>2 || current>6 || next>6 || index>7 || index<1) throw std::runtime_error("Invalid binary game state");
    g.status=status;g.current=current;g.next=next;g.bag.index=index;
    unsigned seen=0;
    for(auto& p:g.bag.pieces) { auto value=r.u32();if(value>6 || (seen&(1U<<value))) throw std::runtime_error("Invalid binary bag");seen|=1U<<value;p=value; }
    g.board.limit=rules==Rules::battleblocks?20:22;
    for(auto& row:g.board.rows) { auto value=r.u32();if(value>full_row) throw std::runtime_error("Invalid binary board");row=value; }
    if(g.board.limit==20 && (g.board.rows[20] || g.board.rows[21])) throw std::runtime_error("Invalid hidden rows");
    g.board.rebuild();
    if (g.lines > (std::numeric_limits<std::uint64_t>::max()-g.board.occupied)/10 ||
        g.pieces > std::numeric_limits<std::uint64_t>::max()/4 || g.lines*10+g.board.occupied != g.pieces*4)
        throw std::runtime_error("Binary game violates cell conservation");
    return g;
}
void put_games(Writer& w,const std::vector<GameState>& games) { w.u64(games.size());for(const auto& g:games) put_game(w,g); }
std::vector<GameState> get_games(Reader& r,Rules rules) { auto n=r.count(164);std::vector<GameState> games;games.reserve(n);for(std::size_t i=0;i<n;++i) games.push_back(get_game(r,rules));return games; }
} // namespace
void validate_config(const Config& c) {
    const auto max=std::numeric_limits<std::uint64_t>::max();
    if(static_cast<unsigned>(c.rules)>2 || c.population<2 || c.population>1000000 || !c.games || c.games>1000000 ||
       !c.runs || c.runs>100000 || c.validation_games>1000000 || c.generations>100000000 || !c.pieces ||
       !c.validation_pieces || !c.checkpoint_pieces || c.pieces>max/4/c.games || c.validation_pieces>max/4/std::max(1U,c.validation_games) ||
       !std::isfinite(c.mutation) || c.mutation<0 || c.mutation>1 || c.sampling>2 || c.seed>max-(c.runs-1))
        throw std::invalid_argument("Invalid experiment bounds");
}
void save_binary(const std::string& path,const Experiment& e) {
    validate_config(e.config);
    Writer w; put_config(w,e.config);w.u64(e.training.size());
    for(const auto& run:e.training) {
        w.u64(run.seed);w.u64(run.random_state);w.u64(run.next_id);w.u32(std::bit_cast<std::uint32_t>(run.generation));
        w.u64(run.population.size());
        for(const auto& c:run.population) { w.u64(c.id);w.weights(c.weights);put_evaluation(w,c.evaluation); }
        w.u64(run.history.size());
        for(const auto& h:run.history) { w.u32(h.generation);w.u64(h.best);w.u64(h.evaluated_pieces);w.f64(h.mean);w.weights(h.weights); }
        put_games(w,run.validation);
    }
    w.u64(e.policies.size());for(const auto& p:e.policies) { w.u64(p.training_seed);w.weights(p.weights);w.u32(p.baseline); }
    put_games(w,e.games);
    if(e.kind==FileKind::statistics) {
        // Summaries are stored after the raw records for simple downstream consumption.
        w.u64(e.policies.size());
        for(std::size_t p=0;p<e.policies.size();++p) {
            auto s=summarize(e.games,p*e.config.games,e.config.games);
            w.u64(s.games);w.u64(s.topouts);w.u64(s.censored);w.u64(s.active);
            w.u32(s.mean_ci_available);
            for(auto v:{s.mean_lines,s.median_lines,s.p05_lines,s.p95_lines,s.sd_lines,s.ci_low,s.ci_high,
                        s.mean_pieces,s.median_pieces,s.p05_pieces,s.p95_pieces}) w.f64(v);
            for(auto milestone:std::array<std::uint64_t,5>{1000,10000,100000,1000000,e.config.pieces}) {
                auto km=survival(e.games,p*e.config.games,e.config.games,milestone);
                w.u64(milestone);w.u32(km.identifiable);w.f64(km.probability);w.f64(km.ci_low);w.f64(km.ci_high);
            }
        }
    }
    if(w.data.size()>maximum_payload) throw std::runtime_error("Binary snapshot exceeds 1 GiB limit");
    auto parent=std::filesystem::path(path).parent_path();if(!parent.empty()) std::filesystem::create_directories(parent);
    Writer header;header.u32(format_version);header.u32(static_cast<std::uint32_t>(e.kind));header.u64(w.data.size());header.u64(checksum(w.data));
    std::ofstream out(path+".tmp",std::ios::binary);out.exceptions(std::ios::failbit|std::ios::badbit);
    out.write("TGC2BIN\0",8);out.write(reinterpret_cast<const char*>(header.data.data()),header.data.size());
    out.write(reinterpret_cast<const char*>(w.data.data()),w.data.size());out.close();
    std::filesystem::rename(path+".tmp",path);
}
Experiment load_binary(const std::string& path) {
    std::ifstream in(path,std::ios::binary);if(!in) throw std::runtime_error("Cannot open "+path);
    in.seekg(0,std::ios::end);auto size=in.tellg();in.seekg(0);
    if(size<32 || size>static_cast<std::streamoff>(maximum_payload+32)) throw std::runtime_error("Invalid binary file size");
    char magic[8];in.read(magic,8);
    if(std::string(magic,8)!=std::string("TGC2BIN\0",8)) throw std::runtime_error("Not a tetris-gc binary file");
    Bytes header(24);in.read(reinterpret_cast<char*>(header.data()),24);Reader hr{header};
    if(hr.u32()!=format_version) throw std::runtime_error("Unsupported binary format version");
    Experiment e;e.kind=static_cast<FileKind>(hr.u32());
    if(static_cast<unsigned>(e.kind)<1 || static_cast<unsigned>(e.kind)>3) throw std::runtime_error("Invalid binary file kind");
    auto length=hr.u64(),hash=hr.u64();if(length!=static_cast<std::uint64_t>(size)-32) throw std::runtime_error("Binary length mismatch");
    Bytes payload(length);in.read(reinterpret_cast<char*>(payload.data()),payload.size());
    if(!in || checksum(payload)!=hash) throw std::runtime_error("Binary checksum mismatch");
    Reader r{payload};e.config=get_config(r);auto count=r.count(52);e.training.reserve(count);
    for(std::size_t i=0;i<count;++i) {
        TrainingRun run;run.seed=r.u64();run.random_state=r.u64();run.next_id=r.u64();run.generation=std::bit_cast<std::int32_t>(r.u32());
        auto n=r.count(72);run.population.reserve(n);
        for(std::size_t j=0;j<n;++j) { Candidate c;c.id=r.u64();c.weights=r.weights();c.evaluation=get_evaluation(r);run.population.push_back(c); }
        n=r.count(60);run.history.reserve(n);
        for(std::size_t j=0;j<n;++j) { History h;h.generation=r.u32();h.best=r.u64();h.evaluated_pieces=r.u64();h.mean=r.f64();h.weights=r.weights();run.history.push_back(h); }
        run.validation=get_games(r,e.config.rules);
        for(const auto& g:run.validation) if(g.pieces>e.config.validation_pieces || (g.status==2 && g.pieces!=e.config.validation_pieces))
            throw std::runtime_error("Invalid validation game cap");
        if(run.population.size()!=e.config.population || run.generation< -1 || run.generation>static_cast<std::int32_t>(e.config.generations) ||
           run.history.size()!=static_cast<std::size_t>(run.generation+1) ||
           (!run.validation.empty() && run.validation.size()!=2ULL*e.config.validation_games)) throw std::runtime_error("Invalid training checkpoint");
        for(std::size_t j=0;j<run.history.size();++j) if(run.history[j].generation!=j) throw std::runtime_error("Invalid history order");
        for(const auto& c:run.population) if(c.id>=run.next_id || c.evaluation.topouts>c.evaluation.games ||
            (run.generation>=0 && c.evaluation.games!=e.config.games)) throw std::runtime_error("Invalid population state");
        e.training.push_back(std::move(run));
    }
    count=r.count(44);e.policies.reserve(count);
    for(std::size_t i=0;i<count;++i) {
        Policy p;p.training_seed=r.u64();p.weights=r.weights();auto flag=r.u32();
        if(flag>1) throw std::runtime_error("Invalid baseline flag");
        p.baseline=flag;e.policies.push_back(p);
    }
    e.games=get_games(r,e.config.rules);
    for(const auto& g:e.games) if(g.pieces>e.config.pieces || (g.status==2 && g.pieces!=e.config.pieces))
        throw std::runtime_error("Invalid endurance game cap");
    if(e.training.size()>e.config.runs || (e.kind==FileKind::training && (!e.policies.empty() || !e.games.empty())) ||
       (e.kind!=FileKind::training && (!e.training.empty() || e.policies.empty() || e.games.size()!=e.policies.size()*e.config.games)))
        throw std::runtime_error("Invalid experiment layout");
    if(e.kind==FileKind::statistics) {
        count=r.count(304);if(count!=e.policies.size()) throw std::runtime_error("Invalid summary count");
        for(std::size_t i=0;i<count;++i) { for(int j=0;j<4;++j) r.u64();if(r.u32()>1) throw std::runtime_error("Invalid confidence flag");for(int j=0;j<11;++j) r.f64();
            for(int j=0;j<5;++j) { r.u64();if(r.u32()>1) throw std::runtime_error("Invalid survival flag");for(int k=0;k<3;++k) r.f64(); } }
    }
    if(r.pos!=payload.size()) throw std::runtime_error("Unexpected binary trailing data");
    return e;
}
Summary summarize(const std::vector<GameState>& games,std::size_t first,std::size_t count) {
    if(!count || first>games.size() || count>games.size()-first) throw std::invalid_argument("Invalid summary range");
    Summary s;s.games=count;std::vector<double> lines,pieces;lines.reserve(count);pieces.reserve(count);
    for(std::size_t i=first;i<first+count;++i) { const auto& g=games[i];lines.push_back(g.lines);pieces.push_back(g.pieces);
        s.topouts+=g.status==1;s.censored+=g.status==2;s.active+=g.status==0; }
    s.mean_lines=std::accumulate(lines.begin(),lines.end(),0.0)/count;
    s.mean_ci_available=count>1 && s.active==0;
    s.mean_pieces=std::accumulate(pieces.begin(),pieces.end(),0.0)/count;
    double variance=0;for(auto v:lines) variance+=(v-s.mean_lines)*(v-s.mean_lines);
    s.sd_lines=count>1?std::sqrt(variance/(count-1)):0;
    double error=1.96*s.sd_lines/std::sqrt(static_cast<double>(count));s.ci_low=s.mean_lines-error;s.ci_high=s.mean_lines+error;
    std::sort(lines.begin(),lines.end());std::sort(pieces.begin(),pieces.end());
    auto quantile=[count](const std::vector<double>& v,double p) { double idx=p*(count-1);auto lo=static_cast<std::size_t>(idx);auto hi=std::min(lo+1,count-1);return v[lo]+(idx-lo)*(v[hi]-v[lo]); };
    s.median_lines=quantile(lines,.5);s.p05_lines=quantile(lines,.05);s.p95_lines=quantile(lines,.95);
    s.median_pieces=quantile(pieces,.5);s.p05_pieces=quantile(pieces,.05);s.p95_pieces=quantile(pieces,.95);return s;
}
Survival survival(const std::vector<GameState>& games,std::size_t first,std::size_t count,std::uint64_t milestone) {
    if(!count || first>games.size() || count>games.size()-first) throw std::invalid_argument("Invalid survival range");
    std::vector<std::pair<std::uint64_t,bool>> observations;observations.reserve(count);
    for(std::size_t i=first;i<first+count;++i) observations.emplace_back(games[i].pieces,games[i].status==1);
    std::sort(observations.begin(),observations.end());
    double probability=1,greenwood=0;std::size_t at_risk=count,idx=0;
    while(idx<count && observations[idx].first<=milestone) {
        auto time=observations[idx].first;std::size_t events=0,end=idx;
        while(end<count && observations[end].first==time) { events+=observations[end].second;++end; }
        if(events) { probability*=1-static_cast<double>(events)/at_risk;
            if(events<at_risk) greenwood+=static_cast<double>(events)/(static_cast<double>(at_risk)*(at_risk-events)); }
        at_risk-=end-idx;idx=end;
    }
    bool identifiable=milestone<=observations.back().first || probability==0;
    // A log-log Greenwood interval remains bounded; endpoints use exact limiting values.
    double low=probability,high=probability;
    if(probability>0 && probability<1) {
        auto se=std::sqrt(greenwood)/std::abs(std::log(probability));auto ll=std::log(-std::log(probability));
        low=std::exp(-std::exp(ll+1.96*se));high=std::exp(-std::exp(ll-1.96*se));
    }
    // Without early censoring, survival at a fixed horizon is a binomial
    // proportion. Wilson intervals also handle zero observed deaths correctly.
    bool fully_observed=true;std::size_t survived=0;
    for(auto [time,event]:observations) {
        if(!event && time<milestone) fully_observed=false;
        survived+=time>milestone || (!event && time==milestone);
    }
    if(fully_observed) {
        double n=static_cast<double>(count),p=static_cast<double>(survived)/n,z2=1.96*1.96;
        double center=(p+z2/(2*n))/(1+z2/n);
        double radius=1.96*std::sqrt(p*(1-p)/n+z2/(4*n*n))/(1+z2/n);
        low=std::max(0.0,center-radius);high=std::min(1.0,center+radius);
    } else if(probability==0 || probability==1) { low=0;high=1; }
    return {identifiable,probability,low,high};
}
} // namespace tetris
