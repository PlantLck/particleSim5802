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

// ============================================================================
// Utility Macros and Timing
// ============================================================================

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

// ============================================================================
// Constant Memory Configuration
// ============================================================================

struct SimulationConstants {
    float friction;
    float restitution;
    float mouse_force;
    int mouse_x;
    int mouse_y;
    bool mouse_pressed;
    bool mouse_attract;
    float dt;
    float width;
    float height;
};

__constant__ SimulationConstants d_constants;

static SimulationConstants cached_constants = {-1, -1, -1, -1, -1, false, false, -1, -1, -1};
static bool constants_initialized = false;

// ============================================================================
// Global Memory Pointers
// ============================================================================

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

// ============================================================================
// Device Helper Functions
// ============================================================================

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
    float max_dist = 150.0f;
    float max_dist_sq = max_dist * max_dist;
    
    if (dist_sq > max_dist_sq || dist_sq < 1e-6f) return;
    
    float dist = sqrtf(dist_sq);
    float force = d_constants.mouse_force * (1.0f - dist / max_dist);
    
    if (!d_constants.mouse_attract) force = -force;
    
    float fx = (dx / dist) * force;
    float fy = (dy / dist) * force;
    
    p->vx += fx / p->mass * dt;
    p->vy += fy / p->mass * dt;
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

// ============================================================================
// Kernel 1: Update Particles (Position Integration)
// ============================================================================

__global__ void update_particles_kernel(Particle* particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    if (!p->active) return;
    
    // Apply friction
    if (d_constants.friction > 0.0f) {
        float friction_factor = 1.0f - d_constants.friction;
        p->vx *= friction_factor;
        p->vy *= friction_factor;
    }
    
    // Apply mouse force
    apply_mouse_force(p, dt);
    
    // Euler integration
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Boundary conditions
    resolve_wall_collision(p);
    limit_velocity(p);
}

// ============================================================================
// Kernel 2: Count Particles Per Cell (Two-Phase with Shared Memory)
// ============================================================================

__global__ void count_particles_per_cell_optimized(
    Particle* particles, int count, int* block_counts, int num_cells) {
    
    extern __shared__ int shared_counts[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    for (int i = tid; i < num_cells; i += blockDim.x) {
        shared_counts[i] = 0;
    }
    __syncthreads();
    
    // Count particles in shared memory
    if (idx < count && particles[idx].active) {
        int cell = get_cell_id(particles[idx].x, particles[idx].y);
        if (cell >= 0 && cell < num_cells) {
            atomicAdd(&shared_counts[cell], 1);
        }
    }
    __syncthreads();
    
    // Write to global memory
    for (int i = tid; i < num_cells; i += blockDim.x) {
        if (shared_counts[i] > 0) {
            block_counts[blockIdx.x * num_cells + i] = shared_counts[i];
        }
    }
}

__global__ void reduce_block_counts_kernel(
    int* block_counts, int* grid_counts, int num_blocks, int num_cells) {
    
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;
    
    int total = 0;
    for (int b = 0; b < num_blocks; b++) {
        total += block_counts[b * num_cells + cell];
    }
    
    atomicAdd(&grid_counts[cell], total);
}

// ============================================================================
// Kernel 3: Prefix Sum (Blelloch Scan)
// ============================================================================

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
    
    // Up-sweep phase
    for (int stride = 1; stride < n_padded; stride *= 2) {
        int threads = n_padded / (2 * stride);
        int blocks = (threads + 255) / 256;
        if (blocks > 0) {
            prefix_sum_up_sweep<<<blocks, 256>>>(d_prefix_sum_temp, n_padded, stride);
        }
    }
    
    CUDA_CHECK(cudaMemset(d_prefix_sum_temp + n_padded - 1, 0, sizeof(int)));
    
    // Down-sweep phase
    for (int stride = n_padded / 2; stride > 0; stride /= 2) {
        int threads = n_padded / (2 * stride);
        int blocks = (threads + 255) / 256;
        if (blocks > 0) {
            prefix_sum_down_sweep<<<blocks, 256>>>(d_prefix_sum_temp, n_padded, stride);
        }
    }
    
    CUDA_CHECK(cudaMemcpy(d_data, d_prefix_sum_temp, n * sizeof(int), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// Kernel 4: Fill Particle Indices
// ============================================================================

__global__ void fill_particle_indices_kernel(
    Particle* particles, int count,
    int* grid_indices,
    int* grid_starts,
    int* grid_counts) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    if (!p->active) return;
    
    int cell = get_cell_id(p->x, p->y);
    int position = atomicAdd(&grid_counts[cell], 1);
    int index = grid_starts[cell] + position;
    
    if (index < count) {
        grid_indices[index] = idx;
    }
}

// ============================================================================
// Kernel 5: Particle Reordering
// ============================================================================

__global__ void reorder_particles_by_cell_kernel(
    Particle* particles, Particle* particles_sorted, int* indices, int count) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int original_idx = indices[idx];
    particles_sorted[idx] = particles[original_idx];
}

__global__ void restore_particle_order_kernel(
    Particle* particles_sorted, Particle* particles, int* indices, int count) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    int original_idx = indices[idx];
    particles[original_idx] = particles_sorted[idx];
}

// ============================================================================
// Kernel 6: Collision Detection (Optimized with Shared Memory)
// ============================================================================

__global__ void detect_collisions_optimized(
    Particle* particles,
    int* grid_indices,
    int* grid_starts,
    int* grid_counts,
    int count) {
    
    extern __shared__ Particle shared_particles[];
    
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
    
    // Check neighboring cells
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = cx + dx;
            int ny = cy + dy;
            
            if (nx < 0 || nx >= grid_w || ny < 0 || ny >= grid_h) continue;
            
            int neighbor_cell = ny * grid_w + nx;
            int start = grid_starts[neighbor_cell];
            int cell_count = grid_counts[neighbor_cell];
            
            // Load particles into shared memory in batches
            for (int batch_start = 0; batch_start < cell_count; batch_start += blockDim.x) {
                int local_idx = threadIdx.x;
                int particle_idx = batch_start + local_idx;
                
                if (particle_idx < cell_count) {
                    int p_idx = grid_indices[start + particle_idx];
                    if (p_idx < count) {
                        shared_particles[local_idx] = particles[p_idx];
                    }
                }
                __syncthreads();
                
                // Check collisions with particles in shared memory
                int batch_size = min(static_cast<int>(blockDim.x), cell_count - batch_start);
                for (int i = 0; i < batch_size; i++) {
                    Particle* p2 = &shared_particles[i];
                    if (!p2->active) continue;
                    
                    if (p1 == p2) continue;
                    
                    float dx_val = p2->x - p1->x;
                    float dy_val = p2->y - p1->y;
                    float dist_sq = dx_val * dx_val + dy_val * dy_val;
                    
                    float min_dist = p1->radius + p2->radius;
                    float min_dist_sq = min_dist * min_dist;
                    
                    if (dist_sq >= min_dist_sq || dist_sq < 1e-6f) continue;
                    
                    // Check if particles are separating
                    float dvx = p2->vx - p1->vx;
                    float dvy = p2->vy - p1->vy;
                    if (dvx * dx_val + dvy * dy_val > 0.0f) continue;
                    
                    float dist = sqrtf(dist_sq);
                    float overlap = min_dist - dist;
                    float total_mass = p1->mass + p2->mass;
                    
                    float sep_x = (dx_val / dist) * overlap * (p2->mass / total_mass);
                    float sep_y = (dy_val / dist) * overlap * (p2->mass / total_mass);
                    
                    p1->x -= sep_x;
                    p1->y -= sep_y;
                    
                    // Calculate and apply impulse
                    float nx = dx_val / dist;
                    float ny = dy_val / dist;
                    float rel_vel_normal = dvx * nx + dvy * ny;
                    float impulse_mag = -(1.0f + d_constants.restitution) * rel_vel_normal / 
                                        (1.0f / p1->mass + 1.0f / p2->mass);
                    
                    float impulse_x = impulse_mag * nx;
                    float impulse_y = impulse_mag * ny;
                    
                    p1->vx -= impulse_x / p1->mass;
                    p1->vy -= impulse_y / p1->mass;
                }
                __syncthreads();
            }
        }
    }
}

// ============================================================================
// Simple Kernels (Baseline Implementation)
// ============================================================================

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
    
    // Brute force O(n²) collision detection
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

// ============================================================================
// Memory Management
// ============================================================================

void init_gpu_memory_optimized(int max_particles) {
    if (gpu_initialized) return;
    
    // Unified memory allocation
    CUDA_CHECK(cudaMallocManaged(&d_particles, max_particles * sizeof(Particle)));
    CUDA_CHECK(cudaMallocManaged(&d_particles_sorted, max_particles * sizeof(Particle)));
    
    // Device memory for grid structures
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

// ============================================================================
// Constants Update
// ============================================================================

void update_constants_if_needed(const Simulation* sim, bool verbose) {
    SimulationConstants new_constants;
    new_constants.friction = sim->get_friction();
    new_constants.restitution = sim->get_restitution();
    new_constants.mouse_force = sim->get_mouse_force();
    new_constants.mouse_x = sim->get_mouse_x();
    new_constants.mouse_y = sim->get_mouse_y();
    new_constants.mouse_pressed = sim->is_mouse_pressed();
    new_constants.mouse_attract = sim->is_mouse_attract();
    new_constants.width = static_cast<float>(sim->get_window_width());
    new_constants.height = static_cast<float>(sim->get_window_height());
    
    if (!constants_initialized || 
        memcmp(&cached_constants, &new_constants, sizeof(SimulationConstants)) != 0) {
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_constants, &new_constants, sizeof(SimulationConstants)));
        cached_constants = new_constants;
        constants_initialized = true;
    }
}

// ============================================================================
// Main GPU Update Functions
// ============================================================================

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
    update_constants_if_needed(sim, verbose);
    
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
    
    CudaTimer timer_constants;
    timer_constants.begin();
    update_constants_if_needed(sim, verbose);
    double constant_copy_ms = timer_constants.end();
    
    // Update particle positions
    CudaTimer timer_update;
    timer_update.begin();
    update_particles_kernel<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    double update_kernel_ms = timer_update.end();
    
    // Two-phase counting with shared memory
    CudaTimer timer_count;
    timer_count.begin();
    
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, NUM_CELLS * sizeof(int)));
    
    int shared_mem_size = NUM_CELLS * sizeof(int);
    count_particles_per_cell_optimized<<<blocks, threads, shared_mem_size>>>(
        d_particles, count, d_block_counts, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    int reduce_blocks = (NUM_CELLS + threads - 1) / threads;
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    double grid_count_ms = timer_count.end();
    
    // GPU-only prefix sum
    CudaTimer timer_prefix;
    timer_prefix.begin();
    
    CUDA_CHECK(cudaMemcpy(d_grid_starts, d_grid_counts, NUM_CELLS * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    
    parallel_prefix_sum_gpu_only(d_grid_starts, NUM_CELLS);
    
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    
    double prefix_sum_ms = timer_prefix.end();
    
    // Fill particle indices
    CudaTimer timer_fill;
    timer_fill.begin();
    
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, NUM_CELLS * sizeof(int)));
    
    fill_particle_indices_kernel<<<blocks, threads>>>(
        d_particles, count, d_grid_indices, d_grid_starts, d_grid_counts);
    CUDA_CHECK(cudaGetLastError());
    
    double grid_fill_ms = timer_fill.end();
    
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    
    // Reorder particles for coalesced access
    CudaTimer timer_reorder;
    timer_reorder.begin();
    
    reorder_particles_by_cell_kernel<<<blocks, threads>>>(
        d_particles, d_particles_sorted, d_grid_indices, count);
    CUDA_CHECK(cudaGetLastError());
    
    double reorder_time = timer_reorder.end();
    
    // Collision detection with shared memory
    CudaTimer timer_collision;
    timer_collision.begin();
    
    int collision_shared_mem = threads * sizeof(Particle);
    detect_collisions_optimized<<<blocks, threads, collision_shared_mem>>>(
        d_particles_sorted, d_grid_indices, d_grid_starts, d_grid_counts, count);
    CUDA_CHECK(cudaGetLastError());
    
    double collision_ms = timer_collision.end();
    
    // Restore original particle order
    CudaTimer timer_restore;
    timer_restore.begin();
    
    restore_particle_order_kernel<<<blocks, threads>>>(
        d_particles_sorted, d_particles, d_grid_indices, count);
    CUDA_CHECK(cudaGetLastError());
    
    double restore_time = timer_restore.end();
    
    // Synchronize and copy back
    double t_sync_start = get_time_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(sim->get_particle_data(), d_particles, count * sizeof(Particle));
    double t_sync_end = get_time_us();
    double d2h_transfer_ms = (t_sync_end - t_sync_start) / 1000.0;
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    if (verbose) {
        printf("\n[GPU COMPLEX - Frame %d]\n", sim->get_frame_counter());
        printf("Particles: %d\n", count);
        printf("Constant copy:    %7.3f ms\n", constant_copy_ms);
        printf("Update kernel:    %7.3f ms\n", update_kernel_ms);
        printf("Grid counting:    %7.3f ms\n", grid_count_ms);
        printf("Prefix sum:       %7.3f ms\n", prefix_sum_ms);
        printf("Fill indices:     %7.3f ms\n", grid_fill_ms);
        printf("Reorder:          %7.3f ms\n", reorder_time);
        printf("Collision:        %7.3f ms\n", collision_ms);
        printf("Restore order:    %7.3f ms\n", restore_time);
        printf("Sync+Transfer:    %7.3f ms\n", d2h_transfer_ms);
        printf("Total time:       %7.3f ms (%.1f FPS)\n", 
               total_time_ms, 1000.0 / total_time_ms);
        printf("\n");
    }
}
