#include "ParticleSimulation.hpp"
#include <SDL.h>
#include <SDL_ttf.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <cstdio>

// ============================================================================
// Graphics class for SDL rendering
// ============================================================================

class Graphics {
private:
    SDL_Window* window;
    SDL_Renderer* renderer;
    TTF_Font* font;
    bool show_menu;
    bool show_stats;
    
public:
    Graphics() : window(nullptr), renderer(nullptr), font(nullptr), 
                 show_menu(true), show_stats(true) {}
    
    ~Graphics() {
        cleanup();
    }
    
    bool initialize() {
        if (SDL_Init(SDL_INIT_VIDEO) < 0) {
            return false;
        }
        
        if (TTF_Init() < 0) {
            return false;
        }
        
        window = SDL_CreateWindow(
            "Parallel Particle Simulation - C++",
            SDL_WINDOWPOS_CENTERED,
            SDL_WINDOWPOS_CENTERED,
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            SDL_WINDOW_SHOWN
        );
        
        if (!window) return false;
        
        renderer = SDL_CreateRenderer(window, -1,
                                      SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
        
        if (!renderer) return false;
        
        // Try to load font from multiple possible locations
        const char* font_paths[] = {
            // Windows paths
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
            // Linux paths
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            nullptr
        };

        for (int i = 0; font_paths[i] != nullptr && !font; i++) {
            font = TTF_OpenFont(font_paths[i], 14);
            if (font) {
                printf("Successfully loaded font: %s\n", font_paths[i]);
                break;
            }
        }

        if (!font) {
            fprintf(stderr, "Warning: Could not load any font. TTF_Error: %s\n", TTF_GetError());
            fprintf(stderr, "Text rendering will not be available.\n");
        }
        
        return true;
    }
    
    void cleanup() {
        if (font) {
            TTF_CloseFont(font);
            font = nullptr;
        }
        if (renderer) {
            SDL_DestroyRenderer(renderer);
            renderer = nullptr;
        }
        if (window) {
            SDL_DestroyWindow(window);
            window = nullptr;
        }
        TTF_Quit();
        SDL_Quit();
    }
    
    void draw_filled_circle(int cx, int cy, int radius) {
        for (int y = -radius; y <= radius; y++) {
            for (int x = -radius; x <= radius; x++) {
                if (x * x + y * y <= radius * radius) {
                    SDL_RenderDrawPoint(renderer, cx + x, cy + y);
                }
            }
        }
    }
    
    void render_text(const std::string& text, int x, int y, SDL_Color color) {
        if (!font || !renderer) return;
        
        SDL_Surface* surface = TTF_RenderText_Solid(font, text.c_str(), color);
        if (!surface) return;
        
        SDL_Texture* texture = SDL_CreateTextureFromSurface(renderer, surface);
        if (texture) {
            SDL_Rect rect = {x, y, surface->w, surface->h};
            SDL_RenderCopy(renderer, texture, nullptr, &rect);
            SDL_DestroyTexture(texture);
        }
        
        SDL_FreeSurface(surface);
    }
    
    void render_particles(Simulation& sim) {
        double start_time = Utils::get_time_ms();
        
        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
        SDL_RenderClear(renderer);
        
        // Draw all particles
        const auto& particles = sim.get_particles();
        for (const auto& p : particles) {
            if (!p.active) continue;
            
            SDL_SetRenderDrawColor(renderer, p.r, p.g, p.b, 255);
            draw_filled_circle(static_cast<int>(p.x), 
                             static_cast<int>(p.y),
                             static_cast<int>(p.radius));
        }
        
        double end_time = Utils::get_time_ms();
        sim.set_render_time(end_time - start_time);
    }
    
    void render_stats(Simulation& sim) {
        if (!show_stats) return;
        
        SDL_Color white = {255, 255, 255, 255};
        SDL_Color green = {100, 255, 100, 255};
        SDL_Color yellow = {255, 255, 100, 255};
        SDL_Color red = {255, 100, 100, 255};
        
        const auto& metrics = sim.get_metrics();
        int y_offset = 10;
        int line_height = 20;
        
        // FPS
        std::ostringstream ss;
        ss << "FPS: " << std::fixed << std::setprecision(1) << metrics.fps;
        render_text(ss.str(), 10, y_offset, green);
        y_offset += line_height;
        
        // Particle count
        ss.str("");
        ss << "Particles: " << sim.get_particle_count();
        render_text(ss.str(), 10, y_offset, white);
        y_offset += line_height;
        
        // Mode
        ss.str("");
        ss << "Mode: " << Utils::get_mode_name(sim.get_mode());
        render_text(ss.str(), 10, y_offset, yellow);
        y_offset += line_height;
        
        // Physics time
        ss.str("");
        ss << "Physics: " << std::fixed << std::setprecision(2) << metrics.physics_time_ms << " ms";
        render_text(ss.str(), 10, y_offset, white);
        y_offset += line_height;
        
        // Render time
        ss.str("");
        ss << "Render: " << std::fixed << std::setprecision(2) << metrics.render_time_ms << " ms";
        render_text(ss.str(), 10, y_offset, white);
        y_offset += line_height;
        
        // Friction
        ss.str("");
        ss << "Friction: " << std::fixed << std::setprecision(4) << sim.get_friction();
        render_text(ss.str(), 10, y_offset, white);
        y_offset += line_height;
        
        // Temperature
        SDL_Color temp_color = white;
        if (metrics.temperature_c > 70.0f) temp_color = red;
        else if (metrics.temperature_c > 55.0f) temp_color = yellow;
        
        ss.str("");
        ss << "Temp: " << std::fixed << std::setprecision(1) << metrics.temperature_c << " C";
        render_text(ss.str(), 10, y_offset, temp_color);
        y_offset += line_height;
        
        // Power
        ss.str("");
        ss << "Power: " << std::fixed << std::setprecision(2) << metrics.power_watts << " W";
        render_text(ss.str(), 10, y_offset, white);
    }
    
    void render_menu(Simulation& sim) {
        if (!show_menu) return;
        
        SDL_Color white = {255, 255, 255, 255};
        SDL_Color highlight = {100, 255, 255, 255};
        
        // Draw semi-transparent background
        SDL_SetRenderDrawBlendMode(renderer, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 180);
        SDL_Rect menu_rect = {WINDOW_WIDTH - 350, 10, 340, 400};
        SDL_RenderFillRect(renderer, &menu_rect);
        
        int x = WINDOW_WIDTH - 340;
        int y = 20;
        int line_height = 25;
        
        render_text("=== CONTROLS ===", x, y, white);
        y += static_cast<int>(line_height * 1.5);
        
        render_text("[M] Toggle Menu", x, y, white);
        y += line_height;
        
        render_text("[SPACE] Pause/Resume", x, y, white);
        y += line_height;
        
        render_text("[R] Reset Simulation", x, y, white);
        y += line_height;
        
        render_text("[+/-] Add/Remove Particles", x, y, white);
        y += line_height;
        
        render_text("[1-5] Change Parallel Mode", x, y, white);
        y += line_height;
        
        render_text("[F/G] Decrease/Increase Friction", x, y, white);
        y += line_height;
        
        render_text("[Mouse Click] Attract Particles", x, y, white);
        y += line_height;
        
        render_text("[Right Click] Repel Particles", x, y, white);
        y += line_height;
        
        render_text("[ESC] Exit", x, y, white);
        y += static_cast<int>(line_height * 1.5);
        
        render_text("=== CURRENT SETTINGS ===", x, y, white);
        y += static_cast<int>(line_height * 1.5);
        
        std::ostringstream ss;
        ss << "Particles: " << sim.get_particle_count() << " / " << sim.get_max_particles();
        render_text(ss.str(), x, y, white);
        y += line_height;
        
        ss.str("");
        ss << "Friction: " << std::fixed << std::setprecision(4) << sim.get_friction();
        render_text(ss.str(), x, y, white);
        y += line_height;
        
        ss.str("");
        ss << "Status: " << (sim.is_running() ? "Running" : "Paused");
        render_text(ss.str(), x, y, sim.is_running() ? highlight : white);
    }
    
    void present() {
        SDL_RenderPresent(renderer);
    }
    
    void toggle_menu() { show_menu = !show_menu; }
    void toggle_stats() { show_stats = !show_stats; }
};

// ============================================================================
// Global Graphics Instance (for main.cpp to access)
// ============================================================================

static std::unique_ptr<Graphics> g_graphics;

extern "C" {
    bool init_graphics() {
        g_graphics = std::make_unique<Graphics>();
        return g_graphics->initialize();
    }
    
    void cleanup_graphics() {
        g_graphics.reset();
    }
    
    void render_frame(Simulation* sim) {
        if (g_graphics && sim) {
            g_graphics->render_particles(*sim);
            g_graphics->render_stats(*sim);
            g_graphics->render_menu(*sim);
            g_graphics->present();
        }
    }
    
    void toggle_menu() {
        if (g_graphics) {
            g_graphics->toggle_menu();
        }
    }
    
    void toggle_stats() {
        if (g_graphics) {
            g_graphics->toggle_stats();
        }
    }
}
