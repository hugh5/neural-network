//
//  problem.hpp
//  neural-network
//
//  Created by Hugh Drummond on 8/6/2025.
//

#ifndef problem_hpp
#define problem_hpp

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include "neural_network.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <memory>


inline void SDL_RenderDrawCircle(SDL_Renderer* ren, int cx, int cy, int radius)
{
    if (!ren || radius <= 0) {                 // trivial cases
        SDL_RenderDrawPoint(ren, cx, cy);
        return;
    }

    auto drawSpan = [&](int x1, int x2, int y) {
        SDL_RenderDrawLine(ren, x1, y, x2, y); // one call draws whole row
    };

    int dx = radius;           // midpoint-circle decision variables
    int dy = 0;
    int d  = 1 - dx;           // start at (r,0)

    while (dx >= dy) {
        // 8-way symmetry: four horizontal spans cover all octants
        drawSpan(cx - dx, cx + dx, cy + dy);   //  E–W on the “north” side
        drawSpan(cx - dx, cx + dx, cy - dy);   //  E–W on the “south” side
        drawSpan(cx - dy, cx + dy, cy + dx);   //  N–S on the “east” side
        drawSpan(cx - dy, cx + dy, cy - dx);   //  N–S on the “west” side

        ++dy;
        if (d < 0) {
            d += (dy << 1) + 1;                // move inside circle
        } else {
            --dx;
            d += ((dy - dx) << 1) + 1;         // move diagonally
        }
    }
}

class Problem {
public:
    virtual ~Problem() = default;
    virtual std::vector<std::vector<double>> getInputs() = 0;  // Made non-const
    virtual std::vector<std::vector<double>> getOutputs() = 0; // Made non-const
    virtual std::vector<size_t> getArchitecture() const = 0;
    virtual double getLearningRate() const = 0;
    virtual double getEpochs() const = 0;
    virtual std::string getName() const = 0;
    virtual void renderPoints(SDL_Renderer* renderer, int x_off, int y_off, int canvas_w, int canvas_h) const {}
};

// XOR Problem
class XORProblem : public Problem {
private:
    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> outputs = {{0}, {1}, {1}, {0}};
    double learning_rate = 0.7;
    int epochs_per_draw = 10;

public:
    std::vector<std::vector<double>> getInputs() override { return inputs; }
    std::vector<std::vector<double>> getOutputs() override { return outputs; }
    std::vector<size_t> getArchitecture() const override { return {2, 8, 8, 1}; }
    double getLearningRate() const override { return learning_rate; }
    double getEpochs() const override { return epochs_per_draw; }
    std::string getName() const override { return "XOR Problem"; }
    
    void renderPoints(SDL_Renderer* renderer, int x_off, int y_off, int canvas_w, int canvas_h) const override {
        // Draw training points
        for (size_t i = 0; i < inputs.size(); ++i) {
            int x = static_cast<int>(inputs[i][0] * canvas_w) + x_off;
            int y = static_cast<int>(inputs[i][1] * canvas_h) + y_off;
            
            if (outputs[i][0] > 0.5) {
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green for 1
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red for 0
            }
            
            // Draw circle
            SDL_RenderDrawCircle(renderer, x, y, 3);
        }
    }
};

// Circle Problem - classify points inside/outside a circle
class CircleProblem : public Problem {
private:
    mutable std::vector<std::vector<double>> cached_inputs;
    mutable std::vector<std::vector<double>> cached_outputs;
    double center_x = 0.5;
    double center_y = 0.5;
    double radius = 0.3;
    int num_points = 100;
    double learning_rate = 0.15;
    int epochs_per_draw = 10;
    
    void generateData() {
        cached_inputs.clear();
        cached_outputs.clear();
        
        // Generate training data
        for (int i = 0; i < num_points; ++i) {
            double x = static_cast<double>(rand()) / RAND_MAX;
            double y = static_cast<double>(rand()) / RAND_MAX;
            
            cached_inputs.push_back({x, y});
            
            // Check if point is inside circle
            double dist = sqrt((x - center_x) * (x - center_x) + (y - center_y) * (y - center_y));
            cached_outputs.push_back({dist <= radius ? 1.0 : 0.0});
        }
    }
public:
    CircleProblem() { }
    std::vector<std::vector<double>> getInputs() override {
        generateData();
        return cached_inputs;
    }
    std::vector<std::vector<double>> getOutputs() override { return cached_outputs; }
    std::vector<size_t> getArchitecture() const override { return {2, 8, 16, 8, 1}; }
    double getLearningRate() const override { return learning_rate; }
    double getEpochs() const override { return epochs_per_draw; }
    std::string getName() const override { return "Circle Classification"; }
    
    void renderPoints(SDL_Renderer* renderer, int x_off, int y_off, int canvas_w, int canvas_h) const override {
        // Draw circle boundary
        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
        int cx = static_cast<int>(center_x * canvas_w) + x_off;
        int cy = static_cast<int>(center_y * canvas_h) + y_off;
        int r = static_cast<int>(radius * canvas_w);
        
        // Simple circle drawing
        for (int angle = 0; angle < 360; angle += 2) {
            double rad = angle * M_PI / 180.0;
            int x = cx + static_cast<int>(r * cos(rad));
            int y = cy + static_cast<int>(r * sin(rad));
            SDL_RenderDrawCircle(renderer, x, y, 1);
        }
        
        // Draw some training points
        for (size_t i = 0; i < std::min(cached_inputs.size(), size_t(50)); ++i) {
            int x = static_cast<int>(cached_inputs[i][0] * canvas_w) + x_off;
            int y = static_cast<int>(cached_inputs[i][1] * canvas_h) + y_off;
            
            if (cached_outputs[i][0] > 0.5) {
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green for inside
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red for outside
            }
            
            SDL_RenderDrawCircle(renderer, x, y, 3);
        }
    }
};

// Spiral Problem - classify points in two interleaved spirals
class SpiralProblem : public Problem {
private:
    std::vector<std::vector<double>> cached_inputs;
    std::vector<std::vector<double>> cached_outputs;
    double learning_rate = 0.35;
    int epochs_per_draw = 20;
    int num_points = 200;
    
    void generateData() {
        if (cached_inputs.size() != 0) {
            return;
        }
        cached_inputs.clear();
        cached_outputs.clear();
        for (int i = 0; i < num_points; ++i) {
            double t = static_cast<double>(i) / num_points * 4 * M_PI;
            double r = t / (4 * M_PI);
            
            // First spiral
            double x1 = 0.5 + r * cos(t) * 0.5;
            double y1 = 0.5 + r * sin(t) * 0.5;
            cached_inputs.push_back({x1, y1});
            cached_outputs.push_back({1.0});
            
            // Second spiral (offset by π)
            double x2 = 0.5 + r * cos(t + M_PI) * 0.5;
            double y2 = 0.5 + r * sin(t + M_PI) * 0.5;
            cached_inputs.push_back({x2, y2});
            cached_outputs.push_back({0.0});
        }
    }

public:
    SpiralProblem() {
        
    }
    
    std::vector<std::vector<double>> getInputs() override {
        generateData();
        return cached_inputs;
    }
    std::vector<std::vector<double>> getOutputs() override { return cached_outputs; }
    std::vector<size_t> getArchitecture() const override { return {2, 8, 8, 1}; }
    double getLearningRate() const override { return learning_rate; }
    double getEpochs() const override { return epochs_per_draw; }
    std::string getName() const override { return "Spiral Classification"; }
    
    void renderPoints(SDL_Renderer* renderer, int x_off, int y_off, int canvas_w, int canvas_h) const override {
        for (size_t i = 0; i < cached_inputs.size(); ++i) {
            int x = static_cast<int>(cached_inputs[i][0] * canvas_w) + x_off;
            int y = static_cast<int>(cached_inputs[i][1] * canvas_h) + y_off;
            
            if (cached_outputs[i][0] > 0.5) {
                SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255); // Green
            } else {
                SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255); // Red
            }
            
            for (int dx = -3; dx <= 3; ++dx) {
                for (int dy = -3; dy <= 3; ++dy) {
                    if (dx*dx + dy*dy <= 9) {
                        SDL_RenderDrawPoint(renderer, x + dx, y + dy);
                    }
                }
            }
        }
    }
};


#endif /* problem_hpp */
