/*
 * CUDA Physics Implementation
 * GPU-accelerated particle simulation with spatial grid optimization
 */

#include "ParticleSimulation.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

#define MAX_VELOCITY 1000.0f
#define SHARED_PARTICLE_BUFFER_SIZE 256

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

inline double get_time_us() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::micro>(duration).count();
}

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

struct SimulationConstants {
    float friction;
    float restitution;
    float mouse_force;
    float mouse_radius;
    int mouse_x;
    int mouse_y;
    bool mouse_pressed;
    bool mouse_attract;
    float dt;
    float width;
    float height;
};

__constant__ SimulationConstants d_constants;

static SimulationConstants cached_constants = {-1, -1, -1, -1, -1, -1, false, false, -1, -1, -1};
static bool constants_initialized = false;

static Particle* d_particles = nullptr;
static Particle* d_particles_sorted = nullptr;
static int* d_grid_indices = nullptr;
static int* d_grid_starts = nullptr;
static int* d_grid_counts = nullptr;
static int* d_block_counts = nullptr;
static int* d_prefix_sum_temp = nullptr;

static size_t total_gpu_memory_bytes = 0;
static bool gpu_initialized = false;

constexpr int NUM_CELLS = (WINDOW_WIDTH / GRID_CELL_SIZE) * (WINDOW_HEIGHT / GRID_CELL_SIZE);

__device__ inline int get_cell_id(float x, float y) {
    int cx = static_cast<int>(x) / GRID_CELL_SIZE;
    int cy = static_cast<int>(y) / GRID_CELL_SIZE;
    int grid_w = WINDOW_WIDTH / GRID_CELL_SIZE;
    int grid_h = WINDOW_HEIGHT / GRID_CELL_SIZE;
    
    if (cx < 0 || cx >= grid_w || cy < 0 || cy >= grid_h) return -1;
    return cy * grid_w + cx;
}

__device__ inline void apply_mouse_force(Particle* p, float dt) {
    if (!d_constants.mouse_pressed) return;
    
    float dx = d_constants.mouse_x - p->x;
    float dy = d_constants.mouse_y - p->y;
    float dist_sq = dx * dx + dy * dy;
    float max_dist = d_constants.mouse_radius;
    float max_dist_sq = max_dist * max_dist;
    
    if (dist_sq > max_dist_sq || dist_sq < 1e-6f) return;
    
    float dist = sqrtf(dist_sq);
    float normalized_dist = dist / max_dist;
    float force = d_constants.mouse_force * (1.0f - normalized_dist * normalized_dist);
    
    if (!d_constants.mouse_attract) force = -force;
    
    float inv_dist = 1.0f / dist;
    float fx = dx * inv_dist * force;
    float fy = dy * inv_dist * force;
    
    p->vx += fx * dt / p->mass;
    p->vy += fy * dt / p->mass;
}

__device__ inline void resolve_wall_collision(Particle* p) {
    if (p->x - p->radius < 0.0f) {
        p->x = p->radius;
        p->vx = fabsf(p->vx) * d_constants.restitution;
    } else if (p->x + p->radius > d_constants.width) {
        p->x = d_constants.width - p->radius;
        p->vx = -fabsf(p->vx) * d_constants.restitution;
    }
    
    if (p->y - p->radius < 0.0f) {
        p->y = p->radius;
        p->vy = fabsf(p->vy) * d_constants.restitution;
    } else if (p->y + p->radius > d_constants.height) {
        p->y = d_constants.height - p->radius;
        p->vy = -fabsf(p->vy) * d_constants.restitution;
    }
}

__device__ inline void limit_velocity(Particle* p) {
    float speed_sq = p->vx * p->vx + p->vy * p->vy;
    float max_sq = MAX_VELOCITY * MAX_VELOCITY;
    
    if (speed_sq > max_sq) {
        float speed = sqrtf(speed_sq);
        float scale = MAX_VELOCITY / speed;
        p->vx *= scale;
        p->vy *= scale;
    }
}

__global__ void update_particles_kernel(Particle* particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    if (!p->active) return;
    
    if (d_constants.friction > 0.0f) {
        float friction_factor = 1.0f - d_constants.friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    apply_mouse_force(p, dt);
    
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    resolve_wall_collision(p);
    limit_velocity(p);
}

__global__ void count_particles_per_cell_kernel(Particle* particles, int count, int* grid_counts, int num_cells) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    if (!particles[idx].active) return;
    
    int cell = get_cell_id(particles[idx].x, particles[idx].y);
    if (cell >= 0 && cell < num_cells) {
        atomicAdd(&grid_counts[cell], 1);
    }
}

__global__ void prefix_sum_up_sweep(int* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (idx + stride * 2 - 1 < n) {
        data[idx + stride * 2 - 1] += data[idx + stride - 1];
    }
}

__global__ void prefix_sum_down_sweep(int* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * stride * 2;
    if (idx + stride * 2 - 1 < n) {
        int temp = data[idx + stride - 1];
        data[idx + stride - 1] = data[idx + stride * 2 - 1];
        data[idx + stride * 2 - 1] += temp;
    }
}

void parallel_prefix_sum_gpu_only(int* d_data, int n) {
    int n_padded = 1;
    while (n_padded < n) n_padded <<= 1;
    
    CUDA_CHECK(cudaMemcpy(d_prefix_sum_temp, d_data, n * sizeof(int), cudaMemcpyDeviceToDevice));
    if (n_padded > n) {
        CUDA_CHECK(cudaMemset(d_prefix_sum_temp + n, 0, (n_padded - n) * sizeof(int)));
    }
    
    for (int stride = 1; stride < n_padded; stride *= 2) {
        int threads = n_padded / (2 * stride);
        int blocks = (threads + 255) / 256;
        if (blocks > 0) {
            prefix_sum_up_sweep<<<blocks, 256>>>(d_prefix_sum_temp, n_padded, stride);
        }
    }
    
    CUDA_CHECK(cudaMemset(d_prefix_sum_temp + n_padded - 1, 0, sizeof(int)));
    
    for (int stride = n_padded / 2; stride > 0; stride /= 2) {
        int threads = n_padded / (2 * stride);
        int blocks = (threads + 255) / 256;
        if (blocks > 0) {
            prefix_sum_down_sweep<<<blocks, 256>>>(d_prefix_sum_temp, n_padded, stride);
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_data, d_prefix_sum_temp, n * sizeof(int), cudaMemcpyDeviceToDevice));
}

__global__ void fill_particle_indices_kernel(
    Particle* particles, int count,
    int* grid_indices,
    int* grid_starts,
    int* grid_counts,
    int num_cells) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    if (!p->active) return;
    
    int cell = get_cell_id(p->x, p->y);
    if (cell >= 0 && cell < num_cells) {
        int position = atomicAdd(&grid_counts[cell], 1);
        int index = grid_starts[cell] + position;
        
        if (index < count) {
            grid_indices[index] = idx;
        }
    }
}

__global__ void detect_collisions_grid_kernel(
    Particle* particles,
    int* grid_indices,
    int* grid_starts,
    int* grid_counts,
    int count,
    int num_cells) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p1 = &particles[idx];
    if (!p1->active) return;
    
    int cell = get_cell_id(p1->x, p1->y);
    if (cell < 0) return;
    
    int grid_w = WINDOW_WIDTH / GRID_CELL_SIZE;
    int grid_h = WINDOW_HEIGHT / GRID_CELL_SIZE;
    int cx = cell % grid_w;
    int cy = cell / grid_w;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            
            if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;
            
            int neighbor_cell = ny * grid_w + nx;
            int start = grid_starts[neighbor_cell];
            int cell_count = grid_counts[neighbor_cell];
            
            for (int i = 0; i < cell_count; i++) {
                int j = grid_indices[start + i];
                if (j <= idx) continue;
                
                Particle* p2 = &particles[j];
                if (!p2->active) continue;
                
                float dx_val = p2->x - p1->x;
                float dy_val = p2->y - p1->y;
                float dist_sq = dx_val * dx_val + dy_val * dy_val;
                
                float min_dist = p1->radius + p2->radius;
                float min_dist_sq = min_dist * min_dist;
                
                if (dist_sq >= min_dist_sq || dist_sq < 1e-6f) continue;
                
                float dvx = p2->vx - p1->vx;
                float dvy = p2->vy - p1->vy;
                if (dvx * dx_val + dvy * dy_val > 0.0f) continue;
                
                float dist = sqrtf(dist_sq);
                float overlap = min_dist - dist;
                float total_mass = p1->mass + p2->mass;
                
                float sep_x = (dx_val / dist) * overlap * 0.5f;
                float sep_y = (dy_val / dist) * overlap * 0.5f;
                
                atomicAdd(&p1->x, -sep_x);
                atomicAdd(&p1->y, -sep_y);
                atomicAdd(&p2->x, sep_x);
                atomicAdd(&p2->y, sep_y);
                
                float nx_norm = dx_val / dist;
                float ny_norm = dy_val / dist;
                float rel_vel_normal = dvx * nx_norm + dvy * ny_norm;
                float impulse_mag = -(1.0f + d_constants.restitution) * rel_vel_normal / 
                                    (1.0f / p1->mass + 1.0f / p2->mass);
                
                float impulse_x = impulse_mag * nx_norm;
                float impulse_y = impulse_mag * ny_norm;
                
                atomicAdd(&p1->vx, -impulse_x / p1->mass);
                atomicAdd(&p1->vy, -impulse_y / p1->mass);
                atomicAdd(&p2->vx, impulse_x / p2->mass);
                atomicAdd(&p2->vy, impulse_y / p2->mass);
            }
        }
    }
}

__global__ void update_particles_simple_kernel(Particle* particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    if (!p->active) return;
    
    if (d_constants.friction > 0.0f) {
        float friction_factor = 1.0f - d_constants.friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    apply_mouse_force(p, dt);
    
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    resolve_wall_collision(p);
    limit_velocity(p);
}

__global__ void detect_collisions_simple_kernel(Particle* particles, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p1 = &particles[idx];
    if (!p1->active) return;
    
    for (int j = idx + 1; j < count; j++) {
        Particle* p2 = &particles[j];
        if (!p2->active) continue;
        
        float dx = p2->x - p1->x;
        float dy = p2->y - p1->y;
        float dist_sq = dx * dx + dy * dy;
        
        float min_dist = p1->radius + p2->radius;
        float min_dist_sq = min_dist * min_dist;
        
        if (dist_sq >= min_dist_sq || dist_sq < 1e-6f) continue;
        
        float dvx = p2->vx - p1->vx;
        float dvy = p2->vy - p1->vy;
        if (dvx * dx + dvy * dy > 0.0f) continue;
        
        float dist = sqrtf(dist_sq);
        float overlap = min_dist - dist;
        float total_mass = p1->mass + p2->mass;
        
        float sep_x = (dx / dist) * overlap * (p2->mass / total_mass);
        float sep_y = (dy / dist) * overlap * (p2->mass / total_mass);
        
        p1->x -= sep_x;
        p1->y -= sep_y;
        p2->x += sep_x;
        p2->y += sep_y;
        
        float nx = dx / dist;
        float ny = dy / dist;
        float rel_vel_normal = dvx * nx + dvy * ny;
        float impulse_mag = -(1.0f + d_constants.restitution) * rel_vel_normal / 
                            (1.0f / p1->mass + 1.0f / p2->mass);
        
        float impulse_x = impulse_mag * nx;
        float impulse_y = impulse_mag * ny;
        
        p1->vx -= impulse_x / p1->mass;
        p1->vy -= impulse_y / p1->mass;
        p2->vx += impulse_x / p2->mass;
        p2->vy += impulse_y / p2->mass;
    }
}

void init_gpu_memory_optimized(int max_particles) {
    if (gpu_initialized) return;
    
    CUDA_CHECK(cudaMallocManaged(&d_particles, max_particles * sizeof(Particle)));
    CUDA_CHECK(cudaMallocManaged(&d_particles_sorted, max_particles * sizeof(Particle)));
    
    CUDA_CHECK(cudaMalloc(&d_grid_indices, max_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_starts, NUM_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_counts, NUM_CELLS * sizeof(int)));
    
    int max_blocks = (max_particles + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_block_counts, max_blocks * NUM_CELLS * sizeof(int)));
    
    int padded_size = 1;
    while (padded_size < NUM_CELLS) padded_size <<= 1;
    CUDA_CHECK(cudaMalloc(&d_prefix_sum_temp, padded_size * sizeof(int)));
    
    total_gpu_memory_bytes = 
        2 * max_particles * sizeof(Particle) +
        max_particles * sizeof(int) +
        NUM_CELLS * 2 * sizeof(int) +
        max_blocks * NUM_CELLS * sizeof(int) +
        padded_size * sizeof(int);
    
    gpu_initialized = true;
    
    printf("[GPU] Allocated %.2f MB GPU memory\n",
           total_gpu_memory_bytes / (1024.0 * 1024.0));
}

extern "C" void cleanup_gpu_memory() {
    if (gpu_initialized) {
        cudaFree(d_particles);
        cudaFree(d_particles_sorted);
        cudaFree(d_grid_indices);
        cudaFree(d_grid_starts);
        cudaFree(d_grid_counts);
        cudaFree(d_block_counts);
        cudaFree(d_prefix_sum_temp);
        gpu_initialized = false;
    }
}

void update_constants_if_needed(const Simulation* sim, float dt, bool verbose) {
    SimulationConstants new_constants;
    new_constants.friction = sim->get_friction();
    new_constants.restitution = sim->get_restitution();
    new_constants.mouse_force = sim->get_mouse_force();
    new_constants.mouse_radius = sim->get_mouse_force_radius();
    new_constants.mouse_x = sim->get_mouse_x();
    new_constants.mouse_y = sim->get_mouse_y();
    new_constants.mouse_pressed = sim->is_mouse_pressed();
    new_constants.mouse_attract = sim->is_mouse_attract();
    new_constants.dt = dt;
    new_constants.width = static_cast<float>(sim->get_window_width());
    new_constants.height = static_cast<float>(sim->get_window_height());
    
    CUDA_CHECK(cudaMemcpyToSymbol(d_constants, &new_constants, sizeof(SimulationConstants)));
    cached_constants = new_constants;
    constants_initialized = true;
}

extern "C" void update_physics_gpu_simple_cuda(Simulation* sim, float dt) {
    double t_start = get_time_us();
    
    const int count = sim->get_particle_count();
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    bool verbose = sim->is_verbose_logging();
    
    if (!gpu_initialized) {
        init_gpu_memory_optimized(sim->get_max_particles());
    }
    
    memcpy(d_particles, sim->get_particle_data(), count * sizeof(Particle));
    update_constants_if_needed(sim, dt, verbose);
    
    update_particles_simple_kernel<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    
    detect_collisions_simple_kernel<<<blocks, threads>>>(d_particles, count);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(sim->get_particle_data(), d_particles, count * sizeof(Particle));
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    if (verbose) {
        printf("[GPU SIMPLE] Particles: %d, Time: %.3f ms (%.1f FPS)\n", 
               count, total_time_ms, 1000.0 / total_time_ms);
    }
}

extern "C" void update_physics_gpu_complex_cuda(Simulation* sim, float dt) {
    double t_start = get_time_us();
    
    const int count = sim->get_particle_count();
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    bool verbose = sim->is_verbose_logging();
    
    if (!gpu_initialized) {
        init_gpu_memory_optimized(sim->get_max_particles());
    }
    
    memcpy(d_particles, sim->get_particle_data(), count * sizeof(Particle));
    update_constants_if_needed(sim, dt, verbose);
    
    update_particles_kernel<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, NUM_CELLS * sizeof(int)));
    
    count_particles_per_cell_kernel<<<blocks, threads>>>(d_particles, count, d_grid_counts, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(d_grid_starts, d_grid_counts, NUM_CELLS * sizeof(int), cudaMemcpyDeviceToDevice));
    parallel_prefix_sum_gpu_only(d_grid_starts, NUM_CELLS);
    
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, NUM_CELLS * sizeof(int)));
    
    fill_particle_indices_kernel<<<blocks, threads>>>(d_particles, count, d_grid_indices, d_grid_starts, d_grid_counts, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    detect_collisions_grid_kernel<<<blocks, threads>>>(d_particles, d_grid_indices, d_grid_starts, d_grid_counts, count, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(sim->get_particle_data(), d_particles, count * sizeof(Particle));
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    if (verbose) {
        printf("[GPU COMPLEX] Particles: %d, Time: %.3f ms (%.1f FPS)\n", 
               count, total_time_ms, 1000.0 / total_time_ms);
    }
}
