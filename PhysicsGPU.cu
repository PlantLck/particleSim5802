/* 
 * CUDA GPU Kernels for Parallel Particle Simulation (C++ Version)
 * WITH DETAILED PERFORMANCE LOGGING
 */

#include "ParticleSimulation.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>

// High-resolution timer
inline double get_time_us() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::micro>(duration).count();
}

// Forward declarations
extern "C" void cleanup_gpu_memory();
extern "C" void init_gpu_memory(int max_particles);

// CUDA error checking macro with detailed logging
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            fprintf(stderr, "  Failed call: %s\n", #call); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// CUDA event timing helper
class CudaTimer {
private:
    cudaEvent_t start, stop;
    bool started;
    
public:
    CudaTimer() : started(false) {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void begin() {
        cudaEventRecord(start);
        started = true;
    }
    
    double end() {
        if (!started) return 0.0;
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        started = false;
        return static_cast<double>(ms);
    }
};

// Device-side constants
__constant__ float d_friction;
__constant__ float d_restitution;
__constant__ float d_mouse_force;
__constant__ int d_mouse_x;
__constant__ int d_mouse_y;
__constant__ bool d_mouse_pressed;
__constant__ bool d_mouse_attract;

// GPU memory pointers
static Particle *d_particles = nullptr;
static int *d_grid_indices = nullptr;
static int *d_grid_starts = nullptr;
static int *d_grid_counts = nullptr;
static int gpu_max_particles = 0;
static bool gpu_initialized = false;

// GPU memory usage tracking
static size_t total_gpu_memory_bytes = 0;

// ============================================================================
// GPU Memory Management
// ============================================================================

extern "C" void cleanup_gpu_memory() {
    if (!gpu_initialized) return;
    
    if (d_particles) cudaFree(d_particles);
    if (d_grid_indices) cudaFree(d_grid_indices);
    if (d_grid_starts) cudaFree(d_grid_starts);
    if (d_grid_counts) cudaFree(d_grid_counts);
    
    d_particles = nullptr;
    d_grid_indices = nullptr;
    d_grid_starts = nullptr;
    d_grid_counts = nullptr;
    gpu_initialized = false;
    total_gpu_memory_bytes = 0;
}

extern "C" void init_gpu_memory(int max_particles) {
    if (gpu_initialized && gpu_max_particles >= max_particles) {
        return;
    }
    
    if (gpu_initialized) {
        cleanup_gpu_memory();
    }
    
    gpu_max_particles = max_particles;
    total_gpu_memory_bytes = 0;
    
    // Allocate particle memory
    size_t particle_size = max_particles * sizeof(Particle);
    CUDA_CHECK(cudaMalloc(&d_particles, particle_size));
    total_gpu_memory_bytes += particle_size;
    
    // Allocate spatial grid memory
    int grid_size = GRID_WIDTH * GRID_HEIGHT;
    size_t grid_indices_size = max_particles * sizeof(int);
    size_t grid_starts_size = grid_size * sizeof(int);
    size_t grid_counts_size = grid_size * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_grid_indices, grid_indices_size));
    CUDA_CHECK(cudaMalloc(&d_grid_starts, grid_starts_size));
    CUDA_CHECK(cudaMalloc(&d_grid_counts, grid_counts_size));
    
    total_gpu_memory_bytes += grid_indices_size + grid_starts_size + grid_counts_size;
    
    gpu_initialized = true;
    
    printf("[GPU] Memory initialized: %.2f MB total\n", total_gpu_memory_bytes / (1024.0 * 1024.0));
    printf("[GPU]   Particles: %.2f MB\n", particle_size / (1024.0 * 1024.0));
    printf("[GPU]   Grid data: %.2f MB\n", 
           (grid_indices_size + grid_starts_size + grid_counts_size) / (1024.0 * 1024.0));
}

extern "C" size_t get_gpu_memory_usage() {
    return total_gpu_memory_bytes;
}

// ============================================================================
// Device Functions
// ============================================================================

__device__ void apply_mouse_force_gpu(Particle *p, int mx, int my, float force, bool attract) {
    float dx = static_cast<float>(mx) - p->x;
    float dy = static_cast<float>(my) - p->y;
    float dist_sq = dx * dx + dy * dy;
    
    if (dist_sq < 100.0f) dist_sq = 100.0f;
    
    float dist = sqrtf(dist_sq);
    float force_magnitude = force / dist_sq;
    
    float fx = (dx / dist) * force_magnitude;
    float fy = (dy / dist) * force_magnitude;
    
    if (!attract) {
        fx = -fx;
        fy = -fy;
    }
    
    p->vx += fx;
    p->vy += fy;
}

__device__ void resolve_wall_collision_gpu(Particle *p, float friction) {
    bool collided = false;
    
    if (p->x - p->radius < 0) {
        p->x = p->radius;
        p->vx = -p->vx;
        collided = true;
    }
    
    if (p->x + p->radius > WINDOW_WIDTH) {
        p->x = WINDOW_WIDTH - p->radius;
        p->vx = -p->vx;
        collided = true;
    }
    
    if (p->y - p->radius < 0) {
        p->y = p->radius;
        p->vy = -p->vy;
        collided = true;
    }
    
    if (p->y + p->radius > WINDOW_HEIGHT) {
        p->y = WINDOW_HEIGHT - p->radius;
        p->vy = -p->vy;
        collided = true;
    }
    
    if (collided && friction > 0.0f) {
        p->vx *= (1.0f - friction);
        p->vy *= (1.0f - friction);
    }
}

__device__ void resolve_particle_collision_gpu(Particle *p1, Particle *p2, float restitution) {
    float dx = p2->x - p1->x;
    float dy = p2->y - p1->y;
    float dist_sq = dx * dx + dy * dy;
    float min_dist = p1->radius + p2->radius;
    float min_dist_sq = min_dist * min_dist;
    
    if (dist_sq >= min_dist_sq || dist_sq < 0.0001f) return;
    
    float dist = sqrtf(dist_sq);
    float nx = dx / dist;
    float ny = dy / dist;
    
    // Separate particles
    float overlap = min_dist - dist;
    float total_mass = p1->mass + p2->mass;
    float separation_1 = overlap * (p2->mass / total_mass);
    float separation_2 = overlap * (p1->mass / total_mass);
    
    p1->x -= nx * separation_1;
    p1->y -= ny * separation_1;
    p2->x += nx * separation_2;
    p2->y += ny * separation_2;
    
    // Calculate relative velocity
    float dvx = p2->vx - p1->vx;
    float dvy = p2->vy - p1->vy;
    float dvn = dvx * nx + dvy * ny;
    
    if (dvn >= 0) return;
    
    // Calculate impulse
    float impulse = -(1.0f + restitution) * dvn / (1.0f / p1->mass + 1.0f / p2->mass);
    float impulse_x = impulse * nx;
    float impulse_y = impulse * ny;
    
    p1->vx -= impulse_x / p1->mass;
    p1->vy -= impulse_y / p1->mass;
    p2->vx += impulse_x / p2->mass;
    p2->vy += impulse_y / p2->mass;
}

// ============================================================================
// Kernel: Update Particles (Simple)
// ============================================================================

__global__ void update_particles_simple(Particle *particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p = &particles[idx];
    if (!p->active) return;
    
    // Apply friction
    if (d_friction > 0.0f) {
        float friction_factor = 1.0f - d_friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    // Apply mouse force
    if (d_mouse_pressed) {
        apply_mouse_force_gpu(p, d_mouse_x, d_mouse_y, d_mouse_force, d_mouse_attract);
    }
    
    // Update position
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Wall collisions
    resolve_wall_collision_gpu(p, d_friction);
    
    // Limit velocity
    float speed_sq = p->vx * p->vx + p->vy * p->vy;
    constexpr float max_velocity = MAX_VELOCITY;
    if (speed_sq > max_velocity * max_velocity) {
        float speed = sqrtf(speed_sq);
        float scale = max_velocity / speed;
        p->vx *= scale;
        p->vy *= scale;
    }
}

__global__ void detect_collisions_simple(Particle *particles, int count, float restitution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p1 = &particles[idx];
    if (!p1->active) return;
    
    // Check against all other particles (brute force)
    for (int j = idx + 1; j < count; j++) {
        Particle *p2 = &particles[j];
        if (!p2->active) continue;
        
        resolve_particle_collision_gpu(p1, p2, restitution);
    }
}

// ============================================================================
// Kernel: Build Spatial Grid
// ============================================================================

__global__ void count_particles_per_cell(Particle *particles, int count, int *grid_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p = &particles[idx];
    if (!p->active) return;
    
    int grid_x = static_cast<int>(p->x / GRID_CELL_SIZE);
    int grid_y = static_cast<int>(p->y / GRID_CELL_SIZE);
    
    if (grid_x >= 0 && grid_x < GRID_WIDTH && grid_y >= 0 && grid_y < GRID_HEIGHT) {
        int cell_idx = grid_y * GRID_WIDTH + grid_x;
        atomicAdd(&grid_counts[cell_idx], 1);
    }
}

__global__ void fill_particle_indices(Particle *particles, int count,
                                      int *grid_indices, int *grid_starts, int *grid_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p = &particles[idx];
    if (!p->active) return;
    
    int grid_x = static_cast<int>(p->x / GRID_CELL_SIZE);
    int grid_y = static_cast<int>(p->y / GRID_CELL_SIZE);
    
    if (grid_x >= 0 && grid_x < GRID_WIDTH && grid_y >= 0 && grid_y < GRID_HEIGHT) {
        int cell_idx = grid_y * GRID_WIDTH + grid_x;
        int pos = atomicAdd(&grid_counts[cell_idx], 1);
        int insert_idx = grid_starts[cell_idx] + pos;
        grid_indices[insert_idx] = idx;
    }
}

// ============================================================================
// Kernel: Complex Collision Detection with Spatial Grid
// ============================================================================

__global__ void detect_collisions_complex(Particle *particles, int count,
                                         int *grid_indices, int *grid_starts,
                                         int *grid_counts, float restitution) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle *p1 = &particles[idx];
    if (!p1->active) return;
    
    // Determine grid range to check
    int min_grid_x = static_cast<int>((p1->x - p1->radius * 4.0f) / GRID_CELL_SIZE);
    int max_grid_x = static_cast<int>((p1->x + p1->radius * 4.0f) / GRID_CELL_SIZE);
    int min_grid_y = static_cast<int>((p1->y - p1->radius * 4.0f) / GRID_CELL_SIZE);
    int max_grid_y = static_cast<int>((p1->y + p1->radius * 4.0f) / GRID_CELL_SIZE);
    
    // Clamp to grid bounds
    if (min_grid_x < 0) min_grid_x = 0;
    if (max_grid_x >= GRID_WIDTH) max_grid_x = GRID_WIDTH - 1;
    if (min_grid_y < 0) min_grid_y = 0;
    if (max_grid_y >= GRID_HEIGHT) max_grid_y = GRID_HEIGHT - 1;
    
    // Check nearby cells
    for (int gy = min_grid_y; gy <= max_grid_y; gy++) {
        for (int gx = min_grid_x; gx <= max_grid_x; gx++) {
            int cell_idx = gy * GRID_WIDTH + gx;
            int start = grid_starts[cell_idx];
            int count_in_cell = grid_counts[cell_idx];
            
            for (int i = 0; i < count_in_cell; i++) {
                int j = grid_indices[start + i];
                if (j <= idx) continue;
                
                Particle *p2 = &particles[j];
                if (!p2->active) continue;
                
                resolve_particle_collision_gpu(p1, p2, restitution);
            }
        }
    }
}

// ============================================================================
// Kernel: Optimized Update with Shared Memory
// ============================================================================

__global__ void update_particles_optimized(Particle *particles, int count, float dt) {
#ifdef JETSON_NANO
    __shared__ Particle s_particles[128];
    const int shared_size = 128;
#else
    __shared__ Particle s_particles[256];
    const int shared_size = 256;
#endif
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_idx = threadIdx.x;
    
    bool is_valid = false;
    if (idx < count && local_idx < shared_size) {
        s_particles[local_idx] = particles[idx];
        is_valid = s_particles[local_idx].active;
    }
    __syncthreads();
    
    if (!is_valid || idx >= count) {
        __syncthreads();
        return;
    }
    
    Particle *p = &s_particles[local_idx];
    
    // Apply friction
    if (d_friction > 0.0f) {
        float friction_factor = 1.0f - d_friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    // Apply mouse force
    if (d_mouse_pressed) {
        apply_mouse_force_gpu(p, d_mouse_x, d_mouse_y, d_mouse_force, d_mouse_attract);
    }
    
    // Update position
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Wall collisions
    resolve_wall_collision_gpu(p, d_friction);
    
    // Limit velocity
    float speed_sq = p->vx * p->vx + p->vy * p->vy;
    constexpr float max_velocity = MAX_VELOCITY;
    if (speed_sq > max_velocity * max_velocity) {
        float speed = sqrtf(speed_sq);
        float scale = max_velocity / speed;
        p->vx *= scale;
        p->vy *= scale;
    }
    
    __syncthreads();
    particles[idx] = *p;
}

// ============================================================================
// Host Functions - GPU Simple Mode WITH DETAILED LOGGING
// ============================================================================

extern "C" void update_physics_gpu_simple_cuda(Simulation* sim, float dt) {
    double t_start = get_time_us();
    
    if (!gpu_initialized) {
        init_gpu_memory(sim->get_max_particles());
    }
    
    DetailedMetrics& metrics = sim->get_metrics_mutable();
    bool verbose = sim->is_verbose_logging();
    
    // Get simulation parameters
    float friction = sim->get_friction();
    float restitution = sim->get_restitution();
    float mouse_force = sim->get_mouse_force();
    int mouse_x = sim->get_mouse_x();
    int mouse_y = sim->get_mouse_y();
    bool mouse_pressed = sim->is_mouse_pressed();
    bool mouse_attract = sim->is_mouse_attract();
    int count = sim->get_particle_count();
    
    // Timing: Constant memory copy
    double t_const_start = get_time_us();
    CUDA_CHECK(cudaMemcpyToSymbol(d_friction, &friction, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_restitution, &restitution, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_force, &mouse_force, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_x, &mouse_x, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_y, &mouse_y, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_pressed, &mouse_pressed, sizeof(bool)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_attract, &mouse_attract, sizeof(bool)));
    double t_const_end = get_time_us();
    metrics.gpu_constant_copy_ms = (t_const_end - t_const_start) / 1000.0;
    
    // Timing: Host to Device transfer
    double t_h2d_start = get_time_us();
    size_t transfer_size = count * sizeof(Particle);
    CUDA_CHECK(cudaMemcpy(d_particles, sim->get_particle_data(),
                         transfer_size, cudaMemcpyHostToDevice));
    double t_h2d_end = get_time_us();
    metrics.gpu_h2d_transfer_ms = (t_h2d_end - t_h2d_start) / 1000.0;
    
    // Launch parameters
#ifdef JETSON_NANO
    int threads = 128;
#else
    int threads = 256;
#endif
    int blocks = (count + threads - 1) / threads;
    
    // Timing: Update kernel
    CudaTimer timer_update;
    timer_update.begin();
    update_particles_simple<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    metrics.gpu_update_kernel_ms = timer_update.end();
    
    // Timing: Collision kernel
    CudaTimer timer_collision;
    timer_collision.begin();
    detect_collisions_simple<<<blocks, threads>>>(d_particles, count, restitution);
    CUDA_CHECK(cudaGetLastError());
    metrics.gpu_collision_kernel_ms = timer_collision.end();
    
    // Timing: Device to Host transfer
    double t_d2h_start = get_time_us();
    CUDA_CHECK(cudaMemcpy(sim->get_particle_data(), d_particles,
                         transfer_size, cudaMemcpyDeviceToHost));
    double t_d2h_end = get_time_us();
    metrics.gpu_d2h_transfer_ms = (t_d2h_end - t_d2h_start) / 1000.0;
    
    // Timing: Final sync
    double t_sync_start = get_time_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_sync_end = get_time_us();
    metrics.gpu_sync_overhead_ms = (t_sync_end - t_sync_start) / 1000.0;
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    metrics.particle_data_size_bytes = transfer_size;
    metrics.gpu_memory_used_bytes = total_gpu_memory_bytes;
    
    if (verbose) {
        printf("\n[GPU SIMPLE - Frame %d]\n", sim->get_frame_counter());
        printf("  Particles: %d (%zu bytes)\n", count, transfer_size);
        printf("  Constant copy:  %6.3f ms\n", metrics.gpu_constant_copy_ms);
        printf("  H2D transfer:   %6.3f ms (%.1f MB/s)\n", 
               metrics.gpu_h2d_transfer_ms,
               (transfer_size / 1024.0 / 1024.0) / (metrics.gpu_h2d_transfer_ms / 1000.0));
        printf("  Update kernel:  %6.3f ms\n", metrics.gpu_update_kernel_ms);
        printf("  Collision kern: %6.3f ms (O(n²) = %d checks)\n", 
               metrics.gpu_collision_kernel_ms, (count * (count-1)) / 2);
        printf("  D2H transfer:   %6.3f ms (%.1f MB/s)\n", 
               metrics.gpu_d2h_transfer_ms,
               (transfer_size / 1024.0 / 1024.0) / (metrics.gpu_d2h_transfer_ms / 1000.0));
        printf("  Sync overhead:  %6.3f ms\n", metrics.gpu_sync_overhead_ms);
        printf("  TOTAL:          %6.3f ms\n", total_time_ms);
        printf("  Breakdown: Compute=%.1f%% Transfer=%.1f%% Overhead=%.1f%%\n",
               100.0 * (metrics.gpu_update_kernel_ms + metrics.gpu_collision_kernel_ms) / total_time_ms,
               100.0 * (metrics.gpu_h2d_transfer_ms + metrics.gpu_d2h_transfer_ms) / total_time_ms,
               100.0 * (metrics.gpu_constant_copy_ms + metrics.gpu_sync_overhead_ms) / total_time_ms);
    }
}

// ============================================================================
// Host Functions - GPU Complex Mode WITH DETAILED LOGGING
// ============================================================================

extern "C" void update_physics_gpu_complex_cuda(Simulation* sim, float dt) {
    double t_start = get_time_us();
    
    if (!gpu_initialized) {
        init_gpu_memory(sim->get_max_particles());
    }
    
    DetailedMetrics& metrics = sim->get_metrics_mutable();
    bool verbose = sim->is_verbose_logging();
    
    // Get simulation parameters
    float friction = sim->get_friction();
    float restitution = sim->get_restitution();
    float mouse_force = sim->get_mouse_force();
    int mouse_x = sim->get_mouse_x();
    int mouse_y = sim->get_mouse_y();
    bool mouse_pressed = sim->is_mouse_pressed();
    bool mouse_attract = sim->is_mouse_attract();
    int count = sim->get_particle_count();
    
    // Timing: Constant memory copy
    double t_const_start = get_time_us();
    CUDA_CHECK(cudaMemcpyToSymbol(d_friction, &friction, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_restitution, &restitution, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_force, &mouse_force, sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_x, &mouse_x, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_y, &mouse_y, sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_pressed, &mouse_pressed, sizeof(bool)));
    CUDA_CHECK(cudaMemcpyToSymbol(d_mouse_attract, &mouse_attract, sizeof(bool)));
    double t_const_end = get_time_us();
    metrics.gpu_constant_copy_ms = (t_const_end - t_const_start) / 1000.0;
    
    // Timing: Host to Device transfer
    double t_h2d_start = get_time_us();
    size_t transfer_size = count * sizeof(Particle);
    CUDA_CHECK(cudaMemcpy(d_particles, sim->get_particle_data(),
                         transfer_size, cudaMemcpyHostToDevice));
    double t_h2d_end = get_time_us();
    metrics.gpu_h2d_transfer_ms = (t_h2d_end - t_h2d_start) / 1000.0;
    
#ifdef JETSON_NANO
    int threads = 128;
#else
    int threads = 256;
#endif
    int blocks = (count + threads - 1) / threads;
    
    // Timing: Update kernel (optimized with shared memory)
    CudaTimer timer_update;
    timer_update.begin();
    update_particles_optimized<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    metrics.gpu_update_kernel_ms = timer_update.end();
    
    // Build spatial grid
    int grid_size = GRID_WIDTH * GRID_HEIGHT;
    
    // Step 1: Clear and count
    CudaTimer timer_grid_count;
    timer_grid_count.begin();
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, grid_size * sizeof(int)));
    count_particles_per_cell<<<blocks, threads>>>(d_particles, count, d_grid_counts);
    CUDA_CHECK(cudaGetLastError());
    metrics.gpu_grid_count_kernel_ms = timer_grid_count.end();
    
    // Step 2: Prefix sum on CPU
    double t_prefix_start = get_time_us();
    std::vector<int> h_grid_counts(grid_size);
    std::vector<int> h_grid_starts(grid_size);
    
    CUDA_CHECK(cudaMemcpy(h_grid_counts.data(), d_grid_counts,
                         grid_size * sizeof(int), cudaMemcpyDeviceToHost));
    
    h_grid_starts[0] = 0;
    for (int i = 1; i < grid_size; i++) {
        h_grid_starts[i] = h_grid_starts[i-1] + h_grid_counts[i-1];
    }
    
    CUDA_CHECK(cudaMemcpy(d_grid_starts, h_grid_starts.data(),
                         grid_size * sizeof(int), cudaMemcpyHostToDevice));
    double t_prefix_end = get_time_us();
    metrics.gpu_prefix_sum_ms = (t_prefix_end - t_prefix_start) / 1000.0;
    
    // Step 3: Fill indices
    CudaTimer timer_grid_fill;
    timer_grid_fill.begin();
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, grid_size * sizeof(int)));
    fill_particle_indices<<<blocks, threads>>>(d_particles, count,
                                               d_grid_indices, d_grid_starts, d_grid_counts);
    CUDA_CHECK(cudaGetLastError());
    metrics.gpu_grid_fill_kernel_ms = timer_grid_fill.end();
    
    // Step 4: Collision detection with spatial grid
    CudaTimer timer_collision;
    timer_collision.begin();
    detect_collisions_complex<<<blocks, threads>>>(d_particles, count,
                                                   d_grid_indices, d_grid_starts,
                                                   d_grid_counts, restitution);
    CUDA_CHECK(cudaGetLastError());
    metrics.gpu_collision_kernel_ms = timer_collision.end();
    
    // Timing: Device to Host transfer
    double t_d2h_start = get_time_us();
    CUDA_CHECK(cudaMemcpy(sim->get_particle_data(), d_particles,
                         transfer_size, cudaMemcpyDeviceToHost));
    double t_d2h_end = get_time_us();
    metrics.gpu_d2h_transfer_ms = (t_d2h_end - t_d2h_start) / 1000.0;
    
    // Timing: Final sync
    double t_sync_start = get_time_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    double t_sync_end = get_time_us();
    metrics.gpu_sync_overhead_ms = (t_sync_end - t_sync_start) / 1000.0;
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    metrics.particle_data_size_bytes = transfer_size;
    metrics.gpu_memory_used_bytes = total_gpu_memory_bytes;
    
    if (verbose) {
        printf("\n[GPU COMPLEX - Frame %d]\n", sim->get_frame_counter());
        printf("  Particles: %d (%zu bytes)\n", count, transfer_size);
        printf("  Constant copy:    %6.3f ms\n", metrics.gpu_constant_copy_ms);
        printf("  H2D transfer:     %6.3f ms (%.1f MB/s)\n", 
               metrics.gpu_h2d_transfer_ms,
               (transfer_size / 1024.0 / 1024.0) / (metrics.gpu_h2d_transfer_ms / 1000.0));
        printf("  Update kernel:    %6.3f ms (shared mem)\n", metrics.gpu_update_kernel_ms);
        printf("  Grid count kern:  %6.3f ms\n", metrics.gpu_grid_count_kernel_ms);
        printf("  Prefix sum (CPU): %6.3f ms\n", metrics.gpu_prefix_sum_ms);
        printf("  Grid fill kern:   %6.3f ms\n", metrics.gpu_grid_fill_kernel_ms);
        printf("  Collision kern:   %6.3f ms (O(n) spatial grid)\n", metrics.gpu_collision_kernel_ms);
        printf("  D2H transfer:     %6.3f ms (%.1f MB/s)\n", 
               metrics.gpu_d2h_transfer_ms,
               (transfer_size / 1024.0 / 1024.0) / (metrics.gpu_d2h_transfer_ms / 1000.0));
        printf("  Sync overhead:    %6.3f ms\n", metrics.gpu_sync_overhead_ms);
        printf("  TOTAL:            %6.3f ms\n", total_time_ms);
        
        double compute_time = metrics.gpu_update_kernel_ms + metrics.gpu_grid_count_kernel_ms + 
                             metrics.gpu_grid_fill_kernel_ms + metrics.gpu_collision_kernel_ms;
        double transfer_time = metrics.gpu_h2d_transfer_ms + metrics.gpu_d2h_transfer_ms;
        double overhead_time = metrics.gpu_constant_copy_ms + metrics.gpu_prefix_sum_ms + 
                              metrics.gpu_sync_overhead_ms;
        
        printf("  Breakdown: Compute=%.1f%% Transfer=%.1f%% Overhead=%.1f%%\n",
               100.0 * compute_time / total_time_ms,
               100.0 * transfer_time / total_time_ms,
               100.0 * overhead_time / total_time_ms);
    }
}
