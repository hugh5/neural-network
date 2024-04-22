//
//  neural_network.hpp
//  neural-network
//
//  Created by Hugh Drummond on 11/4/2024.
//

#ifndef neural_network_hpp
#define neural_network_hpp

#include "matrix.hpp"
#include <vector>
#include <random>

class NeuralNetwork {
private:
    std::vector<int> sizes;
    std::vector<Matrix> weights;
    std::vector<Matrix> activations;
    
    static double sigmoid(double x);
    static double dsigmoid(double x);
    

public:
    NeuralNetwork(int inputNodes, std::vector<int> hiddenLayerSizes, int outputNodes);
    Matrix feedForward(std::vector<double> input);
    void backPropogate(Matrix result, std::vector<double> expected, double learningRate);
    void train(std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs, double learningRate);
    static std::vector<int> randomOrder(size_t n);
    
    friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& network);
};

#endif /* neural_network_hpp */
