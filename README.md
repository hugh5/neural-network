# Neural Network Visualizer

A real-time neural network training visualizer built in C++ that demonstrates how neural networks learn to solve classification problems through interactive visual feedback.

## Features

- **Real-time Learning Visualization**: Watch decision boundaries evolve as the network trains, showing how AI learns complex patterns
- **Custom Neural Network Implementation**: Built from scratch a complete neural network implementation in C++ with custom matrix operations, backpropagation, and gradient descent - no external ML libraries
- **Demonstrates Classic ML Problems**: XOR logic gates, circle classification, and interleaved spiral separation
- **Solves the XOR Problem: Demonstrates non-linear classification that stumped early AI researchers, requiring hidden layers to solve
- **Performance Metrics**: Live display of training epochs, error rates, and improvement metrics

## Demo

### XOR Problem
![XOR Demo](media/xor_demo.gif)

A classic non-linearly separable problem that requires hidden layers to solve.

### Circle Classification
![Circle Demo](media/circle_demo.gif)

Classifies points as inside or outside a circle, demonstrating boundary learning.

### Spiral Classification
![Spiral Demo](media/spiral_demo.gif)

The most challenging problem featuring two interleaved spirals.

## Technical Implementation

### Neural Network Architecture
- **Feedforward Network**: Fully connected layers with sigmoid activation
- **Backpropagation**: Custom implementation of gradient descent
- **Matrix Operations**: Efficient matrix class for neural network computations
- **Configurable Architecture**: Easy to modify layer sizes and learning rates

### Key Components
- `NeuralNetwork`: Core neural network implementation with training algorithms
- `Matrix`: Custom matrix class optimized for neural network operations
- `Problem`: Abstract base class for different classification problems
- `NeuralVis`: SDL2-based visualization engine

### Supported Problems
1. **XOR Problem** (2-8-8-1 architecture)
   - Learning Rate: 0.7
   - Classic logic gate problem

2. **Circle Classification** (2-8-16-8-1 architecture)
   - Learning Rate: 0.15
   - Boundary detection problem

3. **Spiral Classification** (2-8-8-1 architecture)
   - Learning Rate: 0.35
   - Complex non-linear classification

## Project Structure

```
neural-network/
├── main.cpp              # Entry point and problem selection
├── neural_network.hpp    # Core neural network implementation
├── matrix.hpp            # Matrix operations for neural computations
├── neural_vis.hpp        # SDL2 visualization engine
├── neural_vis.cpp        # Visualization implementation
├── problem.hpp           # Problem definitions and rendering
└── README.md            # This file
```

## Algorithm Details

### Forward Propagation
1. Input layer receives 2D coordinates (x, y)
2. Hidden layers apply weighted sums with sigmoid activation
3. Output layer produces classification probability

### Backpropagation
1. Calculate output error using mean squared error
2. Propagate error backwards through network layers
3. Update weights and biases using gradient descent
4. Repeat for each training example

### Visualization
- **Decision Boundary**: Background color intensity shows network confidence
- **Training Points**: Colored dots show actual classification targets
- **Real-time Metrics**: Display current epoch, error rate, and improvement

## Educational Value

This project demonstrates:
- **Neural Network Fundamentals**: Forward/backward propagation implementation
- **Gradient Descent**: Weight optimization through error minimization
- **Non-linear Classification**: How hidden layers enable complex decision boundaries
- **Real-time Learning**: Visual feedback of the training process
- **C++ System Programming**: Custom data structures and graphics programming

## Building and Running

### Prerequisites
- C++17 compatible compiler
- SDL2 development libraries
- SDL2_ttf for text rendering
- macOS (for font path, easily adaptable to other platforms)

### Build Instructions
```bash
# Install dependencies (macOS with Homebrew)
brew install sdl2 sdl2_ttf

# Clone the repository
git clone https://github.com/yourusername/neural-network-visualizer.git
cd neural-network-visualizer

# Compile
g++ -std=c++17 -I/opt/homebrew/include -L/opt/homebrew/lib \
    -lSDL2 -lSDL2_ttf -framework Accelerate \
    main.cpp neural_vis.cpp -o neural_vis

# Run
./neural_vis
```

### Controls
- **Spacebar**: Start/Stop training
- **ESC**: Exit application

## Future Enhancements

- [ ] Additional activation functions (ReLU, tanh)
- [ ] Momentum and adaptive learning rates
- [ ] Batch processing optimization
- [ ] More complex problem types
- [ ] Network architecture visualization
- [ ] Training data export/import
