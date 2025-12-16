#include "ParticleSimulation.hpp"
#include <SDL.h>
#include <iostream>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

// External rendering functions from Rendering.cpp
extern "C" {
    bool init_graphics();
    void cleanup_graphics();
    void render_frame(Simulation* sim);
    void toggle_menu();
    void toggle_stats();
}

// ============================================================================
// Input Handling
// ============================================================================

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
                
            case SDLK_f:
                sim.adjust_friction(-0.0005f);
                std::cout << "Friction: " << sim.get_friction() << "\n";
                break;
                
            case SDLK_g:
                sim.adjust_friction(0.0005f);
                std::cout << "Friction: " << sim.get_friction() << "\n";
                break;
                
            case SDLK_s:
                toggle_stats();
                break;
        }
    }
    
    static void handle_mouse(const SDL_Event& event, Simulation& sim) {
        switch (event.type) {
            case SDL_MOUSEBUTTONDOWN:
                if (event.button.button == SDL_BUTTON_LEFT) {
                    sim.set_mouse_state(event.button.x, event.button.y, true, true);
                } else if (event.button.button == SDL_BUTTON_RIGHT) {
                    sim.set_mouse_state(event.button.x, event.button.y, true, false);
                }
                break;
                
            case SDL_MOUSEBUTTONUP:
                sim.set_mouse_state(event.button.x, event.button.y, false, true);
                break;
                
            case SDL_MOUSEMOTION:
                sim.set_mouse_state(event.motion.x, event.motion.y,
                                   sim.is_mouse_pressed(), sim.is_mouse_attract());
                break;
        }
    }
};

// ============================================================================
// Main Application
// ============================================================================

class Application {
private:
    Simulation sim;
    bool quit;
    std::chrono::high_resolution_clock::time_point last_time;
    std::chrono::high_resolution_clock::time_point fps_timer;
    int frame_count;
    
public:
    Application(int particle_count) 
        : sim(particle_count), quit(false), frame_count(0) {
        last_time = std::chrono::high_resolution_clock::now();
        fps_timer = last_time;
    }
    
    void print_info() {
        std::cout << "Parallel Particle Simulation - C++\n";
        std::cout << "====================================\n";
        std::cout << "Platform: ";
#ifdef PLATFORM_WINDOWS
        std::cout << "Windows Desktop (RTX 3050)\n";
        std::cout << "Max Particles: " << MAX_PARTICLES << "\n";
#elif defined(PLATFORM_JETSON)
        std::cout << "NVIDIA Jetson\n";
        std::cout << "Max Particles: " << MAX_PARTICLES << "\n";
#else
        std::cout << "Linux Desktop\n";
        std::cout << "Max Particles: " << MAX_PARTICLES << "\n";
#endif
        std::cout << "Initial particles: " << sim.get_particle_count() << "\n";
        std::cout << "Press M to toggle menu\n";
        std::cout << "Press ESC to exit\n\n";
    }
    
    int run() {
        print_info();
        
        while (!quit) {
            // Handle events
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT) {
                    quit = true;
                } else if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
                    InputHandler::handle_keyboard(event.key, sim, quit);
                } else if (event.type == SDL_MOUSEBUTTONDOWN || 
                          event.type == SDL_MOUSEBUTTONUP || 
                          event.type == SDL_MOUSEMOTION) {
                    InputHandler::handle_mouse(event, sim);
                }
            }
            
            // Update physics
            auto current_time = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            // Cap dt to prevent instability
            if (dt > 0.05f) dt = 0.05f;
            
            sim.update(dt);
            
            // Render
            render_frame(&sim);
            
            // Calculate FPS
            frame_count++;
            auto fps_elapsed = std::chrono::duration<float>(current_time - fps_timer).count();
            if (fps_elapsed >= 1.0f) {
                sim.set_fps(frame_count / fps_elapsed);
                frame_count = 0;
                fps_timer = current_time;
            }
            
            // Update system metrics periodically
            SystemMonitor::update_metrics(sim);
        }
        
        std::cout << "\nSimulation terminated successfully\n";
        return 0;
    }
};

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char* argv[]) {
    try {
#ifdef USE_MPI
        MPI_Init(&argc, &argv);
        
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        
        if (rank == 0) {
#endif
            // Initialize graphics
            if (!init_graphics()) {
                std::cerr << "Failed to initialize graphics\n";
#ifdef USE_MPI
                MPI_Abort(MPI_COMM_WORLD, 1);
#endif
                return 1;
            }
            
            // Run simulation
            Application app(DEFAULT_PARTICLE_COUNT);
            int result = app.run();
            
            // Cleanup
            cleanup_graphics();
            
#ifdef USE_MPI
        } else {
            // Non-graphics ranks participate in MPI computations
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
                
                // Check if rank 0 wants to terminate (simplified)
                // In production, would use proper MPI termination signal
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
