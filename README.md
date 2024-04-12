# Neural Network

## Description
This project implements a neural network for classification tasks. It provides a flexible and customizable framework for training and evaluating neural networks on various datasets.
Written in C++, the project provides unoptimized implementations of the neural network architecture, backpropagation algorithm, and optimization techniques. The project is designed to be educational and easy to understand, making it suitable for beginners who want to learn about neural networks and deep learning.

## Features
- Multi-layer perceptron architecture
- Support for different activation functions (e.g., sigmoid, ReLU)
- Backpropagation algorithm for training
- Mini-batch gradient descent optimization
- Cross-entropy loss function
- Regularization techniques (e.g., L2 regularization)
- Dropout regularization
- Easy-to-use API for model creation, training, and evaluation

## XOR Classification Example

The following code snippet demonstrates how to create a neural network model, train it on the XOR dataset, and evaluate its performance.

```cpp
#include "neural_network.h"

int main() {
    srand((unsigned)time(NULL));
    int inputNodes = 2;
    std::vector<int> hiddenLayerSizes = {8, 8};
    int outputNodes = 1;
    NeuralNetwork network(inputNodes, hiddenLayerSizes, outputNodes);
        
    std::vector<std::vector<double>> inputs( {{0, 0}, {0, 1}, {1, 0}, {1, 1}} );
    std::vector<std::vector<double>> outputs( {{0}, {1}, {1}, {0}} );
    double learningRate = 0.1;
    network.train(inputs, outputs, learningRate);
    std::cout << network << std::endl;

    for (int i = 0; i < inputs.size(); ++i) {
        std::cout << network.feedForward(inputs[i]) << std::endl;
    }
    
    return 0;
}
```

Example output:
```
Input: [0, 0], Output: [0.002312], Expected: [0]
Input: [0, 1], Output: [0.999983], Expected: [1]
Input: [1, 0], Output: [0.984202], Expected: [1]
Input: [1, 1], Output: [0.015086], Expected: [0]
```

Neural Network Architecture:
```
Neural Network:
  Layer Sizes: 2 -> 8 -> 8 -> 1
  Weights:
    Input Layer to Layer 1 (last column is bias):
| -0.625167 0.147978 -1.854169 |
| -0.520373 -6.109410 -2.146492 |
| 0.065946 1.315157 -2.103537 |
| 67.132578 -58.841079 -1.559796 |
| 0.119602 1.275017 -2.192054 |
| 60.721829 -52.557729 -1.561684 |
| 21.261541 18.853511 -17.473507 |
| -192.100493 206.452343 -8.920780 |

    Layer 1 to Layer 2  (last column is bias):
| 0.137569 -0.566637 -0.703963 -6.747726 0.140288 -7.117362 13.931150 -7.698649 0.083292 |
| -1.851208 0.222969 1.477496 4.495154 0.228193 5.174373 -11.973248 2.799900 -2.024887 |
| -1.219001 -0.549006 -0.718055 2.434078 -0.517637 -0.064154 -2.361457 0.633892 -1.737328 |
| 0.333656 0.257918 -0.793426 -0.142955 2.117319 1.918867 -3.843097 1.440009 -0.849963 |
| -1.203817 -1.148578 -0.064952 -7.913046 -0.768680 -8.560919 17.074344 -8.971691 -0.170793 |
| 0.304743 0.576414 -1.031199 4.654582 -0.465309 4.566832 -13.488617 4.031362 -1.851032 |
| -0.754264 -0.276401 -0.116121 6.719684 0.053953 5.707353 -18.859614 7.064161 -2.626101 |
| -0.957187 -0.465993 -0.859886 -6.692644 0.084186 -4.125306 9.890288 -6.435340 1.374595 |

    Layer 2 to Ouput Layer (last column is bias):
| 4.474515 -4.375819 -1.083414 -2.010570 5.593890 -3.955553 -4.367189 3.400599 -1.701552 |
```
