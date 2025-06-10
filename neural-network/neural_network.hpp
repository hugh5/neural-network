//
//  neural_network.hpp
//  neural-network
//
//  Created by Hugh Drummond on 11/4/2024.
//

#ifndef neural_network_hpp
#define neural_network_hpp

#include "matrix.hpp"
#include <Accelerate/Accelerate.h>
#include <vector>
#include <iostream>
#include <cmath>
#include <cassert>
#include <random>
#include <algorithm>
#include <numeric>
#include <utility>

class NeuralNetwork {
private:
    std::vector<size_t> architecture;
    std::vector<Matrix> weights;
    std::vector<Matrix> biases;
    double learningRate;
    int totalEpochs;
    std::pair<int, double> prev_error;
    std::pair<int, double> cached_error;
    
    static double sigmoid(double x) {
        return 1.0 / (1.0 + std::exp(-x));
    }
    static double dsigmoid(double x) {
        double s = sigmoid(x);
        return s * (1.0 - s);
    }
    
    std::pair<std::vector<Matrix>, std::vector<Matrix>> forwardPropagate(const Matrix& input) const {
        std::vector<Matrix> activations;
        std::vector<Matrix> zValues; // Store z values for backprop
        
        Matrix currentActivation = input;
        activations.push_back(currentActivation);
        
        for (size_t i = 0; i < weights.size(); ++i) {
            Matrix z = weights[i] * currentActivation + biases[i];
            zValues.push_back(z);
            currentActivation = z.apply(sigmoid);
            activations.push_back(currentActivation);
        }
        
        return std::make_pair(activations, zValues);
    }
    

public:
    // Constructor: takes vector of layer sizes (including input and output)
    NeuralNetwork(const std::vector<size_t>& layers, double lr = 0.5)
    : architecture(layers), learningRate(lr), totalEpochs(0),
    prev_error(std::make_pair(0, 0.0)), cached_error(std::make_pair(0, 0.0)) {
        if (layers.size() < 2) {
            throw std::invalid_argument("Neural network must have at least input and output layers");
        }
        
        // Initialize weights and biases
        for (size_t i = 1; i < layers.size(); ++i) {
            // Weight matrix: current layer size × previous layer size
            Matrix w(layers[i], layers[i-1]);
            w.randomize(-2.0, 2.0);
            weights.push_back(w);
            
            // Bias vector: current layer size × 1
            Matrix b(layers[i], 1);
            b.randomize(-1.0, 1.0);
            biases.push_back(b);
        }
    }
    
    // Prediction methods
    std::vector<double> predict(const std::vector<double>& input) const {
        if (input.size() != architecture[0]) {
            throw std::invalid_argument("Input size must match network input layer");
        }
        // Convert input to matrix
        Matrix inputMatrix(input.size(), 1);
        for (size_t i = 0; i < input.size(); ++i) {
            inputMatrix(i, 0) = input[i];
        }
        
        auto result = forwardPropagate(inputMatrix);
        return result.first.back().toVector();
    }
    
    // Training methods
    void trainSingle(const std::vector<double>& input, const std::vector<double>& target) {
            assert(input.size() == architecture[0] && "Input size must match network input layer");
            assert(target.size() == architecture.back() && "Target size must match network output layer");
            
            // Convert input to matrix
            Matrix inputMatrix(input.size(), 1);
            for (size_t i = 0; i < input.size(); ++i) {
                inputMatrix(i, 0) = input[i];
            }
            
            // Convert target to matrix
            Matrix targetMatrix(target.size(), 1);
            for (size_t i = 0; i < target.size(); ++i) {
                targetMatrix(i, 0) = target[i];
            }
            
            // Forward propagation
            auto forwardResult = forwardPropagate(inputMatrix);
            std::vector<Matrix> activations = forwardResult.first;
            std::vector<Matrix> zValues = forwardResult.second;
            
            // Backward propagation
            std::vector<Matrix> deltas;
            deltas.resize(weights.size());
            
            // Calculate output layer delta (error * sigmoid derivative)
            Matrix outputError = activations.back() - targetMatrix;
            Matrix sigmoidDeriv = zValues.back().apply(dsigmoid);
            deltas[deltas.size() - 1] = outputError.hadamard(sigmoidDeriv);
            
            // Calculate hidden layer deltas (backpropagate)
            for (int i = (int)(weights.size()) - 2; i >= 0; --i) {
                Matrix error = weights[i + 1].transpose() * deltas[i + 1];
                Matrix sigmoidDeriv = zValues[i].apply(dsigmoid);
                deltas[i] = error.hadamard(sigmoidDeriv);
            }
            
            // Update weights and biases
            for (size_t i = 0; i < weights.size(); ++i) {
                // Calculate gradients
                Matrix weightGradient = deltas[i] * activations[i].transpose();
                Matrix biasGradient = deltas[i];
                
                // Update weights and biases
                weights[i] = weights[i] - (weightGradient * learningRate);
                biases[i] = biases[i] - (biasGradient * learningRate);
            }
        }
        
        // Train on batch of data
        void train(const std::vector<std::vector<double>>& inputs,
                   const std::vector<std::vector<double>>& targets,
                   int epochs = 1000,
                   bool shuffle = true) {
            
            assert(inputs.size() == targets.size() && "Number of inputs must match number of targets");
            
            std::vector<size_t> indices(inputs.size());
            std::iota(indices.begin(), indices.end(), 0);
            
            for (int epoch = 0; epoch < epochs; ++epoch) {
                if (shuffle) {
                    std::random_device rd;
                    std::mt19937 g(rd());
                    std::shuffle(indices.begin(), indices.end(), g);
                }
                
                for (size_t idx : indices) {
                    trainSingle(inputs[idx], targets[idx]);
                }
                
                // Print progress every 100 epochs
                totalEpochs++;
            }
            double totalError = 0.0;
            for (size_t i = 0; i < inputs.size(); ++i) {
                std::vector<double> prediction = predict(inputs[i]);
                for (size_t j = 0; j < prediction.size(); ++j) {
                    double error = prediction[j] - targets[i][j];
                    totalError += error * error;
                }
            }
            prev_error = cached_error;
            cached_error = std::make_pair(totalEpochs, totalError / inputs.size());
            std::cout << "Epoch " << totalEpochs << ", Average Error: "
                      << totalError / inputs.size() << std::endl;
        }
    
    // Utility methods
    // Get network architecture
    const std::vector<size_t>& getArchitecture() const {
        return architecture;
    }
    
    // Set learning rate
    void setLearningRate(double lr) {
        learningRate = lr;
    }
    
    std::pair<std::pair<int, double>, std::pair<int, double>> getError() {
        return std::make_pair(cached_error, prev_error);
    }
    
    std::string toString() const {
        std::ostringstream oss;
        
        // Build architecture string
        oss << "Architecture: ";
        for (size_t i = 0; i < architecture.size(); ++i) {
            oss << architecture[i];
            if (i < architecture.size() - 1) {
                oss << " -> ";
            }
        }
        
        // Add learning rate with 2 decimal places
        oss << ". Learning Rate: " << std::fixed << std::setprecision(2) << learningRate;
        
        return oss.str();
    }
    
    friend std::ostream& operator<<(std::ostream& os, const NeuralNetwork& network) {
        
        os << "Neural Network:" << std::endl;
        os << "  Architecture: ";
        for (size_t i = 0; i < network.architecture.size(); ++i) {
            os << network.architecture[i];
            if (i != network.architecture.size() - 1) {
                os << " -> ";
            }
        }
        os << std::endl;
        
        os << "  Weights:" << std::endl;
        for (size_t i = 0; i < network.weights.size(); ++i) {
            os << "    Layer " << i << " to Layer " << i + 1 << ":" << std::endl;
            os << network.weights[i] << std::endl;
        }

        return os;
    }
};

#endif /* neural_network_hpp */
