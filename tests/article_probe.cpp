#include "tetris.hpp"
#include <iostream>
int main() {
    int piece,next;
    while(std::cin >> piece >> next) {
        tetris::Weights w;for(auto& v:w) std::cin >> v;
        tetris::Board b;for(auto& row:b.rows) std::cin >> row;b.rebuild();
        auto p=tetris::best_move(b,piece,next,w,tetris::Rules::article);
        if(!p.valid()) std::cout << "null\n";
        else {
            b.place(p.shape,p);
            for(int y=0;y<22;++y) std::cout << (y?" ":"") << b.rows[y];
            std::cout << '\n';
        }
    }
}
