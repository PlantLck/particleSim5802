#include "ParticleSimulation.hpp"
#include <SDL.h>
#include <iostream>
#include <chrono>

#ifdef USE_MPI
#include <mpi.h>
#endif

extern "C" {
    bool init_graphics();
    void cleanup_graphics();
    void render_frame(Simulation* sim);
    void toggle_menu();
    void toggle_stats();
}

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
            case SDLK_UP:
                sim.adjust_mouse_force(MOUSE_FORCE_STEP);
                std::cout << "Mouse force: " << sim.get_mouse_force() << std::endl;
                break;
            case SDLK_DOWN:
                sim.adjust_mouse_force(-MOUSE_FORCE_STEP);
                std::cout << "Mouse force: " << sim.get_mouse_force() << std::endl;
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
#ifdef USE_MPI
                sim.set_mode(ParallelMode::MPI);
                std::cout << "Switched to MPI mode\n";
#else
                std::cout << "MPI mode not available (not compiled with MPI support)\n";
#endif
                break;
            case SDLK_4:
#ifdef USE_CUDA
                sim.set_mode(ParallelMode::GPU_SIMPLE);
                std::cout << "Switched to GPU Simple mode\n";
#else
                std::cout << "GPU mode not available (not compiled with CUDA support)\n";
#endif
                break;
            case SDLK_5:
#ifdef USE_CUDA
                sim.set_mode(ParallelMode::GPU_COMPLEX);
                std::cout << "Switched to GPU Complex mode\n";
#else
                std::cout << "GPU mode not available (not compiled with CUDA support)\n";
#endif
                break;
        }
    }
    
    static void handle_mouse_button(const SDL_MouseButtonEvent& event, Simulation& sim) {
        if (event.type == SDL_MOUSEBUTTONDOWN) {
            bool attract = (event.button == SDL_BUTTON_LEFT);
            sim.set_mouse_state(event.x, event.y, true, attract);
        } else if (event.type == SDL_MOUSEBUTTONUP) {
            sim.set_mouse_state(sim.get_mouse_x(), sim.get_mouse_y(), false, sim.is_mouse_attract());
        }
    }
    
    static void handle_mouse_motion(const SDL_MouseMotionEvent& event, Simulation& sim) {
        if (sim.is_mouse_pressed()) {
            sim.update_mouse_position(event.x, event.y);
        }
    }
    
    static void handle_mouse_wheel(const SDL_MouseWheelEvent& event, Simulation& sim) {
        if (event.y > 0) {
            sim.adjust_mouse_force(MOUSE_FORCE_STEP);
        } else if (event.y < 0) {
            sim.adjust_mouse_force(-MOUSE_FORCE_STEP);
        }
        std::cout << "Mouse force: " << sim.get_mouse_force() << std::endl;
    }
};

class Application {
private:
    Simulation sim;
    bool quit;
    double fps_accumulator;
    int fps_frame_count;
    double last_fps_update;
    
public:
    Application(int particle_count) : sim(particle_count), quit(false),
                                      fps_accumulator(0.0), fps_frame_count(0),
                                      last_fps_update(0.0) {}
    
    int run() {
        auto last_time = std::chrono::high_resolution_clock::now();
        last_fps_update = Utils::get_time_ms();
        
        while (!quit) {
            auto current_time = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float>(current_time - last_time).count();
            last_time = current_time;
            
            if (dt > 0.05f) dt = 0.05f;
            
            SDL_Event event;
            while (SDL_PollEvent(&event)) {
                switch (event.type) {
                    case SDL_QUIT:
                        quit = true;
                        break;
                    case SDL_KEYDOWN:
                    case SDL_KEYUP:
                        InputHandler::handle_keyboard(event.key, sim, quit);
                        break;
                    case SDL_MOUSEBUTTONDOWN:
                    case SDL_MOUSEBUTTONUP:
                        InputHandler::handle_mouse_button(event.button, sim);
                        break;
                    case SDL_MOUSEMOTION:
                        InputHandler::handle_mouse_motion(event.motion, sim);
                        break;
                    case SDL_MOUSEWHEEL:
                        InputHandler::handle_mouse_wheel(event.wheel, sim);
                        break;
                }
            }
            
            double frame_start = Utils::get_time_ms();
            
            sim.update(dt);
            
            SystemMonitor::update_metrics(sim);
            
            render_frame(&sim);
            
            double frame_end = Utils::get_time_ms();
            sim.set_frame_time(frame_end - frame_start);
            
            fps_frame_count++;
            double current_ms = Utils::get_time_ms();
            if (current_ms - last_fps_update >= 500.0) {
                double elapsed = (current_ms - last_fps_update) / 1000.0;
                double fps = fps_frame_count / elapsed;
                sim.set_fps(fps);
                fps_frame_count = 0;
                last_fps_update = current_ms;
            }
        }
        
        std::cout << "Simulation terminated successfully\n";
        return 0;
    }
};

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
            app.run();
            cleanup_graphics();
            
#ifdef USE_MPI
        } else {
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
