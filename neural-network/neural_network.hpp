//
//  neural_network.hpp
//  neural-network
//
//  Created by Hugh Drummond on 11/4/2024.
//

#ifndef neural_network_hpp
#define neural_network_hpp

#include "matrix.hpp"

class NeuralNetwork {
private:
    int inputNodes;
    int hiddenNodes;
    int outputNodes;
    double learningRate;
    Matrix weightsInputHidden;
    Matrix weightsHiddenOutput;

public:
    NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate);
    // Other methods will be added later
};

#endif /* neural_network_hpp */
