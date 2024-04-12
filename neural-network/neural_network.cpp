//
//  neural_network.cpp
//  neural-network
//
//  Created by Hugh Drummond on 11/4/2024.
//

#include "neural_network.hpp"

NeuralNetwork::NeuralNetwork(int inputNodes, std::vector<int> hiddenLayerSizes, int outputNodes)
    : sizes({inputNodes}) {
        sizes.insert(sizes.end(), hiddenLayerSizes.begin(), hiddenLayerSizes.end());
        sizes.push_back(outputNodes);
        
        for (size_t i = 0; i < sizes.size() - 1; ++i) {
            Matrix weightMatrix(sizes[i + 1], sizes[i] + 1);
            weightMatrix.randomize();
            weights.push_back(weightMatrix);
        }
}

double NeuralNetwork::sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

double NeuralNetwork::dsigmoid(double x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

std::vector<int> NeuralNetwork::randomOrder(size_t n) {
    std::vector<int> numbers(n);
    for (int i = 0; i < n; ++i) {
        numbers[i] = i;
    }
    std::vector<int>& arr = numbers;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = (unsigned)arr.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        int j = dist(gen);
        std::swap(arr[i], arr[j]);
    }
    return numbers;
}

/**
 * Performs feedforward propagation to calculate the output of the neural network given input values.
 * Precondtions:
 *  - input.size() == this.inputNodes
 * Parameters:
 *  - values: A vector containing the input values to be fed into the neural network.
 * Returns:
 *  - Matrix: The output of the neural network after performing feedforward propagation. (n x 1)
 */
Matrix NeuralNetwork::feedForward(std::vector<double> input) {
    if (input.size() != sizes.front()) {
        std::cerr << "ERROR: feedforward input.size() != this.inputNodes" << std::endl;
        return Matrix();
    }
    // Create n x 1 input matrix
    std::vector<std::vector<double>> data({});
    for (int i = 0; i < input.size(); ++i) {
        data.push_back(std::vector<double>({input[i]}));
    }
    
    activations.clear();
    // Calculate activations through forward feed
    Matrix activation(data);
    for (int i = 0; i < weights.size(); ++i) {
        activation.appendRow(1); // Bias
        activations.push_back(activation);
//        std::cout << "\tLayer " << i << " activation" << std::endl << activation << std::endl;
        activation = weights[i] * activation;
        activation.map(sigmoid);
    }
    activations.push_back(activation);

    return activation;
}

void NeuralNetwork::backPropogate(Matrix result,
                                  std::vector<double> expected, double learningRate) {
    if (expected.size() != sizes.back()) {
        std::cerr << "ERROR: feedforward expected.size() != this.outputNodes" << std::endl;
        return;
    }
    // Calculate Errors
    std::vector<std::vector<double>> data({});
    for (int i = 0; i < expected.size(); ++i) {
        data.push_back(std::vector<double>({expected[i]}));
    }
    std::vector<Matrix> errors({});
    Matrix error = Matrix(data) - result;
    errors.push_back(error);
//    std::cout << "\tError 0:\n" << error << std::endl;
    
    for (int i = 1; i < weights.size(); ++i) {
        Matrix weight = weights[weights.size()-i].transpose();
        weight.popRow(); // Bias
        error = weight * error;
        errors.push_back(error);
//        std::cout << "\tError " << i << ":\n" << error << std::endl;
        
    }
    
    // Calculate Gradiate
    for (int i = 0; i < errors.size(); ++i) {
        Matrix row = activations[activations.size()-1-i];
        if (i != 0) {
            row.popRow(); // Bias
        }
        row.map(dsigmoid);
        row = row.elemntWiseMultiply(errors[i]);
        row.scalarMultiply(learningRate);
        // Update Bias
        for (int j = 0; j < row.numRows(); ++j) {
            int k = weights[weights.size()-1-i].numCols() - 1;
            weights[weights.size()-1-i](j,k) += row(j,0);
        }
        
        // Update Weights
        Matrix activation = activations[activations.size()-2-i];
        activation.popRow(); // Bias
        row = row * (activation.transpose());
        for (int j = 0; j < row.numRows(); ++j) {
            for (int k = 0; k < row.numCols(); ++ k) {
                weights[weights.size()-1-i](j,k) += row(j,k);
            }
        }
//        std::cout << row << std::endl;
    }
    
    return;
}

void NeuralNetwork::train(std::vector<std::vector<double>> inputs,
                          std::vector<std::vector<double>> outputs, double learningRate) {
    int epoch = 10000;
    for (int e = 0; e < epoch; ++e) {
        std::vector<int> order = randomOrder(inputs.size());
        for (int i = 0; i < inputs.size(); ++i) {
            Matrix result = feedForward(inputs[order[i]]);
            backPropogate(result, outputs[order[i]], learningRate);
        }
    }
}

std::ostream& operator<<(std::ostream& os, const NeuralNetwork& network) {
    
    os << "Neural Network:" << std::endl;
    os << "  Layer Sizes: ";
    for (size_t i = 0; i < network.sizes.size(); ++i) {
        os << network.sizes[i];
        if (i != network.sizes.size() - 1) {
            os << " -> ";
        }
    }
    os << std::endl;
    
    os << "  Weights:" << std::endl;
    for (size_t i = 0; i < network.weights.size(); ++i) {
        os << "    Layer " << i << " to Layer " << i + 1 << ":" << std::endl;
        os << network.weights[i] << std::endl;
    }
    
    if (!network.activations.empty()) {
        os << "  Activations:" << std::endl;
    }
    for (size_t i = 0; i < network.activations.size(); ++i) {
        os << "    Layer " << i << ":" << std::endl;
        os << network.activations[i] << std::endl;
    }
    
    return os;
}
