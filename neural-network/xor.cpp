//
//  xor.cpp
//  neural-network
//
//  Created by Hugh Drummond on 12/4/2024.
//

#include "xor.hpp"

#define WIDTH 1000
#define HEIGHT 1000
#define RESOLUTION 10
#define COLS WIDTH/RESOLUTION
#define ROWS HEIGHT/RESOLUTION

const std::vector<std::vector<double>> inputs({ {0, 0}, {0, 1}, {1, 0}, {1, 1} });
const std::vector<std::vector<double>> outputs({ {0}, {1}, {1}, {0} });

void render(SDL_Renderer *renderer, NeuralNetwork *network) {
    SDL_RenderClear(renderer);
    
    int x_off = WIDTH * 0.05;
    int y_off = HEIGHT * 0.05;
    SDL_Rect canvas;
    canvas.w = WIDTH - 2*x_off;
    canvas.h = HEIGHT - 2*y_off;
    canvas.x = x_off;
    canvas.y = y_off;
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, SDL_ALPHA_OPAQUE);
    SDL_RenderFillRect(renderer, &canvas);
    
    for (int i = 0; i < 200; ++i) {
        std::vector<int> order = NeuralNetwork::randomOrder(inputs.size());
        for (int j = 0; j < order.size(); ++j) {
            Matrix result = network->feedForward(inputs[order[j]]);
            network->backPropogate(result, outputs[order[j]], 0.2);
        }
    }
    double cols = canvas.w/RESOLUTION;
    double rows = canvas.h/RESOLUTION;
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            double i0 = i / cols;
            double i1 = j / rows;
            Matrix output = network->feedForward({i0, i1});
            double y = output(0, 0);
            SDL_Rect rect;
            rect.x = i * RESOLUTION + x_off;
            rect.y = j * RESOLUTION + y_off;
            rect.w = RESOLUTION;
            rect.h = RESOLUTION;
            SDL_SetRenderDrawColor(renderer, 0xff, 0xff, 0xff, y * 255);
            SDL_RenderFillRect(renderer, &rect);
        }
    }
    
    SDL_SetRenderDrawColor(renderer, 0x11, 0x11, 0x11, SDL_ALPHA_OPAQUE); // Background
    SDL_RenderPresent(renderer);
    return;
}

int xor_vis(void) {
    SDL_Window *window;
    SDL_Renderer *renderer;
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "ERROR: Init Video");
    }
    window = SDL_CreateWindow(
        "XOR",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        1000,
        1000,
        SDL_WINDOW_SHOWN
    );
    if (!window) {
        fprintf(stderr, "ERROR: !window");
    }
    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer) {
        fprintf(stderr, "ERROR: !renderer");
    }
    SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);

    NeuralNetwork network(2, {4, 4}, 1);
    std::cout << network;
    
    bool quit = false;
    SDL_Event event;
    while (!quit) {
        while(SDL_PollEvent(&event)) {
            switch(event.type) {
                case SDL_QUIT:
                    quit = true;
                    break;
            }
        }
        render(renderer, &network);

    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    
    return 0;
}
