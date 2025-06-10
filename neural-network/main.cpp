//
//  main.cpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#include "neural_vis.hpp"

int main(int argc, const char * argv[]) {
//    NerualVis vis(std::make_unique<XORProblem>());
//    NerualVis vis(std::make_unique<CircleProblem>());
    NerualVis vis(std::make_unique<SpiralProblem>());
    if (!vis.init()) {
        return -1;
    }
    vis.run();
    
    return 0;
}
