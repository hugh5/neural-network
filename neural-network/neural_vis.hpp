//
//  neural_vis.hpp
//  neural-network
//
//  Created by Hugh Drummond on 8/6/2025.
//

#include "neural_network.hpp"
#include "problem.hpp"

#ifndef neural_vis_hpp
#define neural_vis_hpp

const int CANVAS_WIDTH = 800;
const int CANVAS_HEIGHT = 800;
const int RESOLUTION = 10;
const int lines_of_text = 3;
const int offset = 50;
const int x_off = offset;
const int y_off_top = offset * lines_of_text;
const int y_off_bottom = offset;

class NerualVis {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    std::unique_ptr<NeuralNetwork> network;
    std::unique_ptr<Problem> problem;

    void render_problem() {
        SDL_SetRenderDrawColor(renderer, 17, 17, 17, 255); // Dark background
        SDL_RenderClear(renderer);
        

        SDL_Rect canvas;
        canvas.w = CANVAS_WIDTH;
        canvas.h = CANVAS_HEIGHT;
        canvas.x = x_off;
        canvas.y = y_off_top;
        
        // Draw canvas background
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderFillRect(renderer, &canvas);

        // Train network
        auto inputs = problem->getInputs();
        auto outputs = problem->getOutputs();
        auto epochs = problem->getEpochs();
        network->train(inputs, outputs, epochs, true);
        
        // Visualize decision boundary
        double cols = canvas.w / RESOLUTION;
        double rows = canvas.h / RESOLUTION;
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                double i0 = static_cast<double>(i) / cols;
                double i1 = static_cast<double>(j) / rows;
                double prediction = network->predict({i0, i1})[0];
                
                SDL_Rect rect;
                rect.x = i * RESOLUTION + x_off;
                rect.y = j * RESOLUTION + y_off_top;
                rect.w = RESOLUTION;
                rect.h = RESOLUTION;
                
                // Color based on prediction
                Uint8 intensity = static_cast<Uint8>(prediction * 255);
                SDL_SetRenderDrawColor(renderer, 255, 255, 255, intensity);
                SDL_RenderFillRect(renderer, &rect);
            }
        }
        auto network_error = network->getError();
        auto current = network_error.first;
        auto prev = network_error.second;
        int epoch = current.first;
        double avg_error = current.second;
        double prev_error = prev.second;
        double improv = prev_error - avg_error;
        renderText(std::format("Epoch: {:4}", epoch),
                   x_off, 0);
        renderText(std::format("Network Error: {:.2f}%. Training Improvment: {:.4f}", avg_error * 100, improv * 100),
                   x_off, 50);
        renderText(network->toString(), x_off, 100);
        
        // Render problem-specific elements (training points, boundaries, etc.)
        problem->renderPoints(renderer, x_off, y_off_top, canvas.w, canvas.h);
        
        SDL_RenderPresent(renderer);
    }
    
    void renderText(const std::string& text, int x, int y, SDL_Color color = {255, 255, 255, 255}) {
            // Create surface from text
            SDL_Surface* textSurface = TTF_RenderText_Solid(font, text.c_str(), color);
            if (textSurface == nullptr) {
                std::cout << "Unable to render text surface! SDL_ttf Error: " << TTF_GetError() << std::endl;
                return;
            }
            
            // Create texture from surface
            SDL_Texture* textTexture = SDL_CreateTextureFromSurface(renderer, textSurface);
            if (textTexture == nullptr) {
                std::cout << "Unable to create texture from rendered text! SDL Error: " << SDL_GetError() << std::endl;
                SDL_FreeSurface(textSurface);
                return;
            }
            
            // Get text dimensions
            int textWidth = textSurface->w;
            int textHeight = textSurface->h;
            SDL_FreeSurface(textSurface);
            
            // Set rendering space and render to screen
            SDL_Rect renderQuad = {x, y, textWidth, textHeight};
            SDL_RenderCopy(renderer, textTexture, nullptr, &renderQuad);
            
            // Clean up
            SDL_DestroyTexture(textTexture);
        }
public:
    
    NerualVis(std::unique_ptr<Problem> prob) {
        problem = std::move(prob);
        // Create neural network based on problem
        network = std::make_unique<NeuralNetwork>(problem->getArchitecture(), problem->getLearningRate());
        
        window = nullptr;
        renderer = nullptr;
        font = nullptr;
    }
    
    bool init() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            std::cerr << "SDL_Init. Error: " << SDL_GetError() << std::endl;
            return false;
        }
        
        if (TTF_Init() == -1) {
            std::cerr << "TTF_Init. Error: " << TTF_GetError() << std::endl;
            return false;
        }
        
        std::string title = "Neural Network Visualizer - " + problem->getName();
        window = SDL_CreateWindow(
            title.c_str(),
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            CANVAS_WIDTH + 2 * x_off,
            CANVAS_HEIGHT + y_off_top + y_off_bottom,
            SDL_WINDOW_SHOWN
        );
        
        if (!window) {
            std::cerr << "SDL_CreateWindow. Error: " << SDL_GetError() << std::endl;
            SDL_Quit();
            return false;
        }
        
        renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
        if (!renderer) {
            std::cerr << "SDL_CreateRenderer. Error: " << SDL_GetError() << std::endl;
            SDL_DestroyWindow(window);
            SDL_Quit();
            return false;
        }
        
        font = TTF_OpenFont("/System/Library/Fonts/SFNSMono.ttf", 24); // macOS
        if (!font) {
            std::cerr << "Failed to load SDL_ttf font. Error: " << TTF_GetError() << std::endl;
            return false;
        }
        
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        return true;
    }
    
    void run() {
        bool quit = false;
        bool train = false;
        SDL_Event event;
        
        std::cout << "Starting visualization for: " << problem->getName() << std::endl;
        std::cout << "Press ESC or close window to quit" << std::endl;
        
        while (!quit) {
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        quit = true;
                        break;
                    case SDL_KEYDOWN:
                        if (event.key.keysym.sym == SDLK_ESCAPE) {
                            quit = true;
                        } else if (event.key.keysym.sym == SDLK_SPACE) {
                            train = !train;
                        }
                        break;
                }
            }
            
            if (train) render_problem();
        }
        
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
    
    void cleanup() {
        // Close font
        if (font != nullptr) {
            TTF_CloseFont(font);
            font = nullptr;
        }
        
        // Destroy renderer and window
        if (renderer != nullptr) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        
        if (window != nullptr) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        
        // Quit SDL subsystems
        TTF_Quit();
        SDL_Quit();
    }
    
    ~NerualVis() {
        cleanup();
    }
};

#endif /* neural_vis_hpp */
