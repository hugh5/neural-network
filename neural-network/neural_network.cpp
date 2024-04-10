//
//  neural_network.cpp
//  neural-network
//
//  Created by Hugh Drummond on 11/4/2024.
//

#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, double learningRate)
    : inputNodes(inputNodes), hiddenNodes(hiddenNodes), outputNodes(outputNodes), learningRate(learningRate) {
    weightsInputHidden = Matrix(hiddenNodes, inputNodes);
    weightsInputHidden.randomize();
    
    weightsHiddenOutput = Matrix(outputNodes, hiddenNodes);
    weightsHiddenOutput.randomize();
}
