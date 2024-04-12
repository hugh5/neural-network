//
//  main.cpp
//  neural-network
//
//  Created by Hugh Drummond on 10/4/2024.
//

#include <iostream>
#include <vector>
#include "matrix.hpp"
#include "neural_network.hpp"

int main(int argc, const char * argv[]) {
    srand((unsigned)time(NULL));
    int inputNodes = 2;
    std::vector<int> hiddenLayerSizes = {8, 8, 8};
    int outputNodes = 1;
    NeuralNetwork network(inputNodes, hiddenLayerSizes, outputNodes);
        
    std::vector<std::vector<double>> inputs( {{0, 0}, {0, 1}, {1,0}, {1,1}} );
    std::vector<std::vector<double>> outputs( {{0}, {1}, {1}, {0}} );
    double learningRate = 0.1;
    network.train(inputs, outputs, learningRate);
    std::cout << network << std::endl;

    for (int i = 0; i < inputs.size(); ++i) {
        std::cout << network.feedForward(inputs[i]) << std::endl;
    }
    
    return 0;
}
