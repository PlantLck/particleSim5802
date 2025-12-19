#include "ParticleSimulation.hpp"
#include <SDL.h>
#include <iostream>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

// External rendering functions
extern "C" {
    bool init_graphics();
    void cleanup_graphics();
    void render_frame(Simulation* sim);
    void toggle_menu();
    void toggle_stats();
}

// Input handling
class InputHandler {
public:
    static void handle_keyboard(const SDL_KeyboardEvent& event, Simulation& sim, bool& quit) {
        if (event.type != SDL_KEYDOWN) return;
        
        switch (event.keysym.sym) {
            case SDLK_ESCAPE:
                quit = true;
                break;
            case SDLK_m:
                toggle_menu();
                break;
            case SDLK_SPACE:
                sim.set_running(!sim.is_running());
                break;
            case SDLK_r:
                sim.request_reset();
                break;
            case SDLK_PLUS:
            case SDLK_EQUALS:
                sim.add_particles(50);
                break;
            case SDLK_MINUS:
                sim.remove_particles(50);
                break;
            case SDLK_1:
                sim.set_mode(ParallelMode::SEQUENTIAL);
                std::cout << "Switched to Sequential mode\n";
                break;
            case SDLK_2:
                sim.set_mode(ParallelMode::MULTITHREADED);
                std::cout << "Switched to Multithreaded mode\n";
                break;
            case SDLK_3:
                sim.set_mode(ParallelMode::MPI);
                std::cout << "Switched to MPI mode\n";
                break;
            case SDLK_4:
                sim.set_mode(ParallelMode::GPU_SIMPLE);
                std::cout << "Switched to GPU Simple mode\n";
                break;
            case SDLK_5:
                sim.set_mode(ParallelMode::GPU_COMPLEX);
                std::cout << "Switched to GPU Complex mode\n";
                break;
        }
    }
    
    static void handle_mouse(const SDL_MouseButtonEvent& event, Simulation& sim) {
        if (event.type == SDL_MOUSEBUTTONDOWN) {
            bool attract = (event.button == SDL_BUTTON_LEFT);
            sim.set_mouse_state(event.x, event.y, true, attract);  // Changed
        } else if (event.type == SDL_MOUSEBUTTONUP) {
            sim.set_mouse_state(0, 0, false, false);  // Changed
        }
    }
};

// Main application
class Application {
private:
    Simulation sim;
    bool quit;
    
public:
    Application(int particle_count) : sim(particle_count), quit(false) {}
    
    int run() {
        auto last_time = std::chrono::high_resolution_clock::now();
        
        while (!quit) {
            auto current_time = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            if (dt > 0.05f) dt = 0.05f;
            
            // Handle events
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = true;
                } else if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
                    InputHandler::handle_keyboard(event.key, sim, quit);
                } else if (event.type == SDL_MOUSEBUTTONDOWN || event.type == SDL_MOUSEBUTTONUP) {
                    InputHandler::handle_mouse(event.button, sim);
                }
            }
            
            // Update and render
            sim.update(dt);
            render_frame(&sim);
        }
        
        std::cout << "Simulation terminated successfully\n";
        return 0;
    }
};

// Main entry point
int main(int argc, char* argv[]) {
    try {
#ifdef USE_MPI
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        if (rank == 0) {
#endif
            if (!init_graphics()) {
                std::cerr << "Failed to initialize graphics\n";
#ifdef USE_MPI
                MPI_Abort(MPI_COMM_WORLD, 1);
#endif
                return 1;
            }
            
            Application app(DEFAULT_PARTICLE_COUNT);
            int result = app.run();
            cleanup_graphics();
            
#ifdef USE_MPI
        } else {
            // Non-graphics ranks for MPI
            Simulation sim(DEFAULT_PARTICLE_COUNT);
            sim.set_mode(ParallelMode::MPI);
            
            bool running = true;
            auto last_time = std::chrono::high_resolution_clock::now();
            
            while (running) {
                auto current_time = std::chrono::high_resolution_clock::now();
                float dt = std::chrono::duration<float>(current_time - last_time).count();
                last_time = current_time;
                
                if (dt > 0.05f) dt = 0.05f;
                PhysicsEngine::update_mpi(sim, dt);
            }
        }
        
        MPI_Finalize();
#endif
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << '\n';
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
    }
}
