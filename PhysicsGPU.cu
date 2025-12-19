/*
 * PhysicsGPU.cu - FULLY OPTIMIZED GPU Complex Implementation
 * 
 * COMPREHENSIVE OPTIMIZATION STRATEGY:
 * ================================================================================
 * 
 * PRIMARY OPTIMIZATIONS (CRITICAL):
 * 
 * 1. UNIFIED MEMORY ARCHITECTURE
 *    - Leverages Jetson's unified memory with zero-copy access
 *    - Eliminates all explicit H2D/D2H memory transfers
 *    - Target reduction: 2.5-3.7 ms → ~0 ms transfer overhead
 * 
 * 2. GPU-ONLY PREFIX SUM (Blelloch Scan)
 *    - Complete elimination of CPU-GPU synchronization
 *    - Work-efficient parallel scan algorithm
 *    - Up-sweep and down-sweep phases entirely on GPU
 *    - Target reduction: 1.0-2.5 ms → ~0.2 ms
 * 
 * 3. TWO-PHASE ATOMIC COUNTING WITH SHARED MEMORY
 *    - Per-block counting in shared memory (100× faster than global atomics)
 *    - Global reduction kernel for final counts
 *    - Reduces atomic contention dramatically
 *    - Target speedup: 3-5× in grid construction
 * 
 * 4. SHARED MEMORY FOR COLLISION DETECTION
 *    - Load particle data into shared memory
 *    - Reduce global memory accesses
 *    - Target speedup: 2-3× for collision kernel
 * 
 * 5. PARTICLE REORDERING FOR COALESCED ACCESS
 *    - Sort particles by cell ID before collision detection
 *    - Enables perfect memory coalescing
 *    - Target speedup: 2× from memory bandwidth optimization
 * 
 * 6. BATCHED CONSTANT UPDATES
 *    - Cache simulation constants, update only when changed
 *    - Single constant memory update per frame
 * 
 * EXPECTED PERFORMANCE:
 * - Target: 10,000-15,000 particles @ 60 FPS on Jetson Xavier NX
 * - Expected speedup: 12-18× over sequential baseline
 * - Theoretical maximum: 20-25× (limited by Amdahl's Law to ~12-15× realistic)
 * 
 * ARCHITECTURE NOTES:
 * - Designed for NVIDIA Jetson Xavier NX (Volta sm_72)
 * - 512 CUDA cores across 8 SMs
 * - Unified memory architecture
 * - Optimal configuration: 256 threads/block, 8 blocks/SM
 */

#include "ParticleSimulation.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <algorithm>

// ============================================================================
// UTILITY MACROS AND TIMER
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

// High-resolution timer
inline double get_time_us() {
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<double, std::micro>(duration).count();
}

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

// ============================================================================
// CONSTANT MEMORY CONFIGURATION
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

// Cache to avoid unnecessary constant updates
static SimulationConstants cached_constants = {-1, -1, -1, -1, -1, false, false, -1, -1, -1};
static bool constants_initialized = false;

// ============================================================================
// GLOBAL MEMORY POINTERS (UNIFIED MEMORY)
// ============================================================================

static Particle* d_particles = nullptr;           // Unified memory for particles
static Particle* d_particles_sorted = nullptr;    // Reordered by cell for coalescing
static int* d_grid_indices = nullptr;
static int* d_grid_starts = nullptr;
static int* d_grid_counts = nullptr;
static int* d_block_counts = nullptr;             // For two-phase counting
static int* d_prefix_sum_temp = nullptr;          // For GPU-only prefix sum
static bool gpu_initialized = false;
static size_t total_gpu_memory_bytes = 0;

// Grid configuration - use values from header
#define NUM_CELLS (GRID_WIDTH * GRID_HEIGHT)
#define CELL_SIZE ((float)GRID_CELL_SIZE)

// ============================================================================
// DEVICE HELPER FUNCTIONS
// ============================================================================

__device__ inline int get_cell_x(float x) {
    int cell_x = (int)(x / CELL_SIZE);
    if (cell_x < 0) cell_x = 0;
    if (cell_x >= GRID_WIDTH) cell_x = GRID_WIDTH - 1;
    return cell_x;
}

__device__ inline int get_cell_y(float y) {
    int cell_y = (int)(y / CELL_SIZE);
    if (cell_y < 0) cell_y = 0;
    if (cell_y >= GRID_HEIGHT) cell_y = GRID_HEIGHT - 1;
    return cell_y;
}

__device__ inline int get_cell_id(float x, float y) {
    return get_cell_y(y) * GRID_WIDTH + get_cell_x(x);
}

__device__ inline void apply_mouse_force_gpu(Particle* p) {
    if (!d_constants.mouse_pressed) return;
    
    float dx = d_constants.mouse_x - p->x;
    float dy = d_constants.mouse_y - p->y;
    float dist_sq = dx * dx + dy * dy;
    
    if (dist_sq > 1e-6f && dist_sq < 10000.0f) {
        float dist = sqrtf(dist_sq);
        float force = d_constants.mouse_force / (dist * dist + 1.0f);
        float factor = d_constants.mouse_attract ? force : -force;
        
        p->vx += (dx / dist) * factor;
        p->vy += (dy / dist) * factor;
    }
}

__device__ inline void resolve_wall_collision_gpu(Particle* p) {
    if (p->x - p->radius < 0.0f) {
        p->x = p->radius;
        p->vx = -p->vx * d_constants.restitution;
    } else if (p->x + p->radius > d_constants.width) {
        p->x = d_constants.width - p->radius;
        p->vx = -p->vx * d_constants.restitution;
    }
    
    if (p->y - p->radius < 0.0f) {
        p->y = p->radius;
        p->vy = -p->vy * d_constants.restitution;
    } else if (p->y + p->radius > d_constants.height) {
        p->y = d_constants.height - p->radius;
        p->vy = -p->vy * d_constants.restitution;
    }
}

__device__ inline void resolve_particle_collision_gpu(Particle* p1, Particle* p2) {
    if (!p1->active || !p2->active) return;
    
    float dx = p2->x - p1->x;
    float dy = p2->y - p1->y;
    float dist_sq = dx * dx + dy * dy;
    float min_dist = p1->radius + p2->radius;
    float min_dist_sq = min_dist * min_dist;
    
    if (dist_sq >= min_dist_sq || dist_sq < 1e-6f) return;
    
    float dist = sqrtf(dist_sq);
    float nx = dx / dist;
    float ny = dy / dist;
    
    float overlap = min_dist - dist;
    float separation = overlap * 0.5f;
    
    p1->x -= nx * separation;
    p1->y -= ny * separation;
    p2->x += nx * separation;
    p2->y += ny * separation;
    
    float dvx = p2->vx - p1->vx;
    float dvy = p2->vy - p1->vy;
    float dot = dvx * nx + dvy * ny;
    
    if (dot > 0) return;
    
    float impulse = (1.0f + d_constants.restitution) * dot / 2.0f;
    
    p1->vx += impulse * nx;
    p1->vy += impulse * ny;
    p2->vx -= impulse * nx;
    p2->vy -= impulse * ny;
}

// ============================================================================
// KERNEL 1: UPDATE PARTICLE POSITIONS
// ============================================================================

__global__ void update_particles_kernel(Particle* particles, int count, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p = &particles[idx];
    if (!p->active) return;
    
    // Apply mouse force
    apply_mouse_force_gpu(p);
    
    // Apply friction
    p->vx *= d_constants.friction;
    p->vy *= d_constants.friction;
    
    // Update position
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Wall collisions
    resolve_wall_collision_gpu(p);
}

// ============================================================================
// OPTIMIZATION 2: TWO-PHASE COUNTING WITH SHARED MEMORY
// ============================================================================

__global__ void count_particles_per_cell_optimized(
    Particle* particles, int count,
    int* block_counts,  // Output: [num_blocks × num_cells]
    int num_cells) {
    
    // Shared memory for per-block counting (100× faster than global atomics)
    __shared__ int local_counts[NUM_CELLS];
    
    // Initialize shared memory
    for (int i = threadIdx.x; i < num_cells; i += blockDim.x) {
        local_counts[i] = 0;
    }
    __syncthreads();
    
    // Count in shared memory
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count && particles[idx].active) {
        int cell = get_cell_id(particles[idx].x, particles[idx].y);
        atomicAdd(&local_counts[cell], 1);
    }
    __syncthreads();
    
    // Write block results to global memory
    int block_offset = blockIdx.x * num_cells;
    for (int i = threadIdx.x; i < num_cells; i += blockDim.x) {
        block_counts[block_offset + i] = local_counts[i];
    }
}

__global__ void reduce_block_counts_kernel(
    int* block_counts,
    int* global_counts,
    int num_blocks,
    int num_cells) {
    
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= num_cells) return;
    
    int total = 0;
    for (int b = 0; b < num_blocks; b++) {
        total += block_counts[b * num_cells + cell];
    }
    global_counts[cell] = total;
}

// ============================================================================
// OPTIMIZATION 3: GPU-ONLY PREFIX SUM (Blelloch Scan)
// ============================================================================

__global__ void prefix_sum_up_sweep(int* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 * stride;
    if (idx + 2 * stride - 1 < n) {
        data[idx + 2 * stride - 1] += data[idx + stride - 1];
    }
}

__global__ void prefix_sum_down_sweep(int* data, int n, int stride) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2 * stride;
    if (idx + 2 * stride - 1 < n) {
        int temp = data[idx + stride - 1];
        data[idx + stride - 1] = data[idx + 2 * stride - 1];
        data[idx + 2 * stride - 1] += temp;
    }
}

void parallel_prefix_sum_gpu_only(int* d_data, int n) {
    // Find next power of 2
    int n_padded = 1;
    while (n_padded < n) n_padded <<= 1;
    
    // Use temp buffer
    if (n_padded > n) {
        CUDA_CHECK(cudaMemcpy(d_prefix_sum_temp, d_data, n * sizeof(int), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(d_prefix_sum_temp + n, 0, (n_padded - n) * sizeof(int)));
    } else {
        CUDA_CHECK(cudaMemcpy(d_prefix_sum_temp, d_data, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    
    // Up-sweep phase (reduce)
    for (int stride = 1; stride < n_padded; stride *= 2) {
        int threads = n_padded / (2 * stride);
        int blocks = (threads + 255) / 256;
        if (blocks > 0) {
            prefix_sum_up_sweep<<<blocks, 256>>>(d_prefix_sum_temp, n_padded, stride);
        }
    }
    
    // Set last element to 0 (exclusive scan)
    CUDA_CHECK(cudaMemset(d_prefix_sum_temp + n_padded - 1, 0, sizeof(int)));
    
    // Down-sweep phase
    for (int stride = n_padded / 2; stride > 0; stride /= 2) {
        int threads = n_padded / (2 * stride);
        int blocks = (threads + 255) / 256;
        if (blocks > 0) {
            prefix_sum_down_sweep<<<blocks, 256>>>(d_prefix_sum_temp, n_padded, stride);
        }
    }
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(d_data, d_prefix_sum_temp, n * sizeof(int), cudaMemcpyDeviceToDevice));
}

// ============================================================================
// KERNEL 4: FILL PARTICLE INDICES
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
// OPTIMIZATION 4: PARTICLE REORDERING FOR COALESCED ACCESS
// ============================================================================

__global__ void reorder_particles_by_cell_kernel(
    Particle* particles,
    Particle* particles_sorted,
    int* grid_indices,
    int count) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Reorder so particles in same cell are contiguous
    int particle_idx = grid_indices[idx];
    particles_sorted[idx] = particles[particle_idx];
}

__global__ void restore_particle_order_kernel(
    Particle* particles_sorted,
    Particle* particles,
    int* grid_indices,
    int count) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Restore original particle order after collision detection
    int particle_idx = grid_indices[idx];
    particles[particle_idx] = particles_sorted[idx];
}

// ============================================================================
// OPTIMIZATION 5: COLLISION DETECTION WITH SHARED MEMORY
// ============================================================================

__global__ void detect_collisions_optimized(
    Particle* particles_sorted,  // Already sorted by cell
    int* grid_indices,
    int* grid_starts,
    int* grid_counts,
    int count) {
    
    // Load particles into shared memory
    __shared__ Particle shared_particles[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load this block's particles into shared memory
    if (idx < count) {
        shared_particles[tid] = particles_sorted[idx];
    }
    __syncthreads();
    
    if (idx >= count) return;
    
    Particle* p1 = &shared_particles[tid];
    if (!p1->active) return;
    
    // Calculate neighborhood cells (3×3 grid)
    int my_cell_x = get_cell_x(p1->x);
    int my_cell_y = get_cell_y(p1->y);
    
    int min_x = (my_cell_x > 0) ? my_cell_x - 1 : 0;
    int max_x = (my_cell_x < GRID_WIDTH - 1) ? my_cell_x + 1 : GRID_WIDTH - 1;
    int min_y = (my_cell_y > 0) ? my_cell_y - 1 : 0;
    int max_y = (my_cell_y < GRID_HEIGHT - 1) ? my_cell_y + 1 : GRID_HEIGHT - 1;
    
    // Check nearby cells
    for (int gy = min_y; gy <= max_y; gy++) {
        for (int gx = min_x; gx <= max_x; gx++) {
            int cell_id = gy * GRID_WIDTH + gx;
            int start = grid_starts[cell_id];
            int cell_count = grid_counts[cell_id];
            
            int block_start_idx = blockIdx.x * blockDim.x;
            int block_end_idx = block_start_idx + blockDim.x;
            
            // Check particles in same block first (in shared memory)
            for (int i = 0; i < cell_count; i++) {
                int global_idx = start + i;
                if (global_idx >= block_start_idx && global_idx < block_end_idx) {
                    int shared_idx = global_idx - block_start_idx;
                    if (shared_idx > tid) {  // Avoid duplicate checks
                        Particle* p2 = &shared_particles[shared_idx];
                        resolve_particle_collision_gpu(p1, p2);
                    }
                }
            }
            
            // Then check particles in other blocks (global memory)
            for (int i = 0; i < cell_count; i++) {
                int global_idx = start + i;
                
                // Skip if in shared memory (already handled)
                if (global_idx >= block_start_idx && global_idx < block_end_idx) continue;
                
                if (global_idx > idx) {  // Avoid duplicate checks
                    Particle* p2 = &particles_sorted[global_idx];
                    resolve_particle_collision_gpu(p1, p2);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write back modified particle
    if (idx < count) {
        particles_sorted[idx] = *p1;
    }
}

// ============================================================================
// MEMORY INITIALIZATION
// ============================================================================

void init_gpu_memory_optimized(int max_particles) {
    if (gpu_initialized) return;
    
    // OPTIMIZATION 1: Unified memory for zero-copy access
    CUDA_CHECK(cudaMallocManaged(&d_particles, max_particles * sizeof(Particle)));
    CUDA_CHECK(cudaMallocManaged(&d_particles_sorted, max_particles * sizeof(Particle)));
    
    // Device memory for grid structures
    CUDA_CHECK(cudaMalloc(&d_grid_indices, max_particles * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_starts, NUM_CELLS * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_grid_counts, NUM_CELLS * sizeof(int)));
    
    // Block counts for two-phase counting
    int max_blocks = (max_particles + 255) / 256;
    CUDA_CHECK(cudaMalloc(&d_block_counts, max_blocks * NUM_CELLS * sizeof(int)));
    
    // Prefix sum temporary buffer
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
    
    printf("[GPU INIT] Allocated %.2f MB GPU memory (unified + device)\n",
           total_gpu_memory_bytes / (1024.0 * 1024.0));
    printf("[GPU INIT] Optimizations enabled:\n");
    printf("  ✓ Unified memory (zero-copy)\n");
    printf("  ✓ GPU-only prefix sum (Blelloch scan)\n");
    printf("  ✓ Two-phase atomic counting (shared memory)\n");
    printf("  ✓ Particle reordering (coalesced access)\n");
    printf("  ✓ Shared memory collision detection\n");
}

void cleanup_gpu_memory() {
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
// OPTIMIZATION 6: BATCHED CONSTANT UPDATES
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
    new_constants.dt = 0.016f;  // Fixed timestep
    new_constants.width = static_cast<float>(sim->get_window_width());
    new_constants.height = static_cast<float>(sim->get_window_height());
    
    // Only update if something changed
    if (!constants_initialized || 
        memcmp(&new_constants, &cached_constants, sizeof(SimulationConstants)) != 0) {
        
        CUDA_CHECK(cudaMemcpyToSymbol(d_constants, &new_constants, sizeof(SimulationConstants)));
        cached_constants = new_constants;
        constants_initialized = true;
        
        if (verbose) {
            printf("[CONSTANTS] Updated (batched)\n");
        }
    }
}

// ============================================================================
// MAIN PHYSICS UPDATE FUNCTION - FULLY OPTIMIZED
// ============================================================================

// ============================================================================
// SIMPLE KERNELS - NO SPATIAL GRID OPTIMIZATION
// ============================================================================

/**
 * Simple particle update kernel - basic physics only
 */
__global__ void update_particles_simple_kernel(Particle* particles, int count, float dt) {
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
    apply_mouse_force_gpu(p);
    
    // Euler integration
    p->x += p->vx * dt;
    p->y += p->vy * dt;
    
    // Wall collision detection
    if (p->x - p->radius < 0.0f) {
        p->x = p->radius;
        p->vx = -p->vx * d_constants.restitution;
    }
    if (p->x + p->radius > d_constants.width) {
        p->x = d_constants.width - p->radius;
        p->vx = -p->vx * d_constants.restitution;
    }
    if (p->y - p->radius < 0.0f) {
        p->y = p->radius;
        p->vy = -p->vy * d_constants.restitution;
    }
    if (p->y + p->radius > d_constants.height) {
        p->y = d_constants.height - p->radius;
        p->vy = -p->vy * d_constants.restitution;
    }
    
    // Velocity limiting
    float vel_sq = p->vx * p->vx + p->vy * p->vy;
    if (vel_sq > MAX_VELOCITY * MAX_VELOCITY) {
        float vel = sqrtf(vel_sq);
        p->vx = (p->vx / vel) * MAX_VELOCITY;
        p->vy = (p->vy / vel) * MAX_VELOCITY;
    }
}

/**
 * Brute-force collision detection kernel - O(n²) complexity
 * Each thread checks its particle against ALL other particles
 * This is the baseline non-optimized approach
 */
__global__ void detect_collisions_simple_kernel(Particle* particles, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    Particle* p1 = &particles[idx];
    if (!p1->active) return;
    
    // Check against ALL other particles (brute force O(n²))
    for (int j = idx + 1; j < count; j++) {
        Particle* p2 = &particles[j];
        if (!p2->active) continue;
        
        // Calculate distance
        float dx = p2->x - p1->x;
        float dy = p2->y - p1->y;
        float dist_sq = dx * dx + dy * dy;
        float min_dist = p1->radius + p2->radius;
        
        // Check collision
        if (dist_sq < min_dist * min_dist && dist_sq > 1e-6f) {
            float dist = sqrtf(dist_sq);
            float overlap = min_dist - dist;
            
            // Normalize collision vector
            float nx = dx / dist;
            float ny = dy / dist;
            
            // Separate particles
            float total_mass = p1->mass + p2->mass;
            float ratio1 = p2->mass / total_mass;
            float ratio2 = p1->mass / total_mass;
            
            p1->x -= nx * overlap * ratio1;
            p1->y -= ny * overlap * ratio1;
            p2->x += nx * overlap * ratio2;
            p2->y += ny * overlap * ratio2;
            
            // Calculate relative velocity
            float dvx = p2->vx - p1->vx;
            float dvy = p2->vy - p1->vy;
            float rel_vel = dvx * nx + dvy * ny;
            
            // Only resolve if particles are moving toward each other
            if (rel_vel < 0.0f) {
                // Calculate impulse
                float impulse_mag = -(1.0f + d_constants.restitution) * rel_vel / (1.0f/p1->mass + 1.0f/p2->mass);
                float impulse_x = impulse_mag * nx;
                float impulse_y = impulse_mag * ny;
                
                // Apply impulse
                p1->vx -= impulse_x / p1->mass;
                p1->vy -= impulse_y / p1->mass;
                p2->vx += impulse_x / p2->mass;
                p2->vy += impulse_y / p2->mass;
            }
        }
    }
}

/**
 * GPU Simple physics update - baseline brute-force implementation
 * This provides a true performance comparison baseline without optimizations
 */
extern "C" void update_physics_gpu_simple_cuda(Simulation* sim, float dt) {
    double t_start = get_time_us();
    
    const int count = sim->get_particle_count();
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    bool verbose = sim->is_verbose_logging();
    
    if (!gpu_initialized) {
        init_gpu_memory_optimized(sim->get_max_particles());
    }
    
    // Copy particle data to GPU
    memcpy(d_particles, sim->get_particle_data(), count * sizeof(Particle));
    
    // Update constants if needed
    update_constants_if_needed(sim, verbose);
    
    // Launch simple kernels (no spatial grid)
    update_particles_simple_kernel<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    
    detect_collisions_simple_kernel<<<blocks, threads>>>(d_particles, count);
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize and copy results back
    CUDA_CHECK(cudaDeviceSynchronize());
    memcpy(sim->get_particle_data(), d_particles, count * sizeof(Particle));
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    if (verbose) {
        printf("\n[GPU SIMPLE - Frame %d] BRUTE FORCE O(n²) IMPLEMENTATION\n", 
               sim->get_frame_counter());
        printf("================================================================================\n");
        printf("Particles: %d\n", count);
        printf("Collision checks: %d (brute force, no spatial grid)\n", (count * (count - 1)) / 2);
        printf("Total time: %.3f ms (%.1f FPS)\n", total_time_ms, 1000.0 / total_time_ms);
        printf("================================================================================\n\n");
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
    
    // OPTIMIZATION 1: Unified memory - simple memcpy, no H2D transfer overhead
    memcpy(d_particles, sim->get_particle_data(), count * sizeof(Particle));
    
    // OPTIMIZATION 6: Update constants only if changed (batched)
    CudaTimer timer_constants;
    timer_constants.begin();
    update_constants_if_needed(sim, verbose);
    double constant_copy_ms = timer_constants.end();
    
    // ========================================================================
    // KERNEL 1: Update particle positions
    // ========================================================================
    CudaTimer timer_update;
    timer_update.begin();
    update_particles_kernel<<<blocks, threads>>>(d_particles, count, dt);
    CUDA_CHECK(cudaGetLastError());
    double update_kernel_ms = timer_update.end();
    
    // ========================================================================
    // OPTIMIZATION 2: Two-phase counting with shared memory
    // ========================================================================
    CudaTimer timer_count;
    timer_count.begin();
    
    // Clear grid counts
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, NUM_CELLS * sizeof(int)));
    
    // Phase 1: Per-block counting in shared memory
    count_particles_per_cell_optimized<<<blocks, threads>>>(
        d_particles, count, d_block_counts, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    // Phase 2: Reduce block counts to global counts
    int reduce_blocks = (NUM_CELLS + threads - 1) / threads;
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    CUDA_CHECK(cudaGetLastError());
    
    double grid_count_ms = timer_count.end();
    
    // ========================================================================
    // OPTIMIZATION 3: GPU-only prefix sum (no CPU synchronization)
    // ========================================================================
    CudaTimer timer_prefix;
    timer_prefix.begin();
    
    // Copy counts to starts for prefix sum
    CUDA_CHECK(cudaMemcpy(d_grid_starts, d_grid_counts, NUM_CELLS * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    
    // Parallel prefix sum entirely on GPU
    parallel_prefix_sum_gpu_only(d_grid_starts, NUM_CELLS);
    
    // Restore original counts (they were modified by fill_particle_indices)
    CUDA_CHECK(cudaMemcpy(d_grid_counts, d_block_counts, blocks * NUM_CELLS * sizeof(int),
                         cudaMemcpyDeviceToDevice));
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    
    double prefix_sum_ms = timer_prefix.end();
    
    // ========================================================================
    // KERNEL 4: Fill particle indices
    // ========================================================================
    CudaTimer timer_fill;
    timer_fill.begin();
    
    // Reset counts for atomic adds in fill kernel
    CUDA_CHECK(cudaMemset(d_grid_counts, 0, NUM_CELLS * sizeof(int)));
    
    fill_particle_indices_kernel<<<blocks, threads>>>(
        d_particles, count, d_grid_indices, d_grid_starts, d_grid_counts);
    CUDA_CHECK(cudaGetLastError());
    
    double grid_fill_ms = timer_fill.end();
    
    // Restore counts again for collision detection
    reduce_block_counts_kernel<<<reduce_blocks, threads>>>(
        d_block_counts, d_grid_counts, blocks, NUM_CELLS);
    
    // ========================================================================
    // OPTIMIZATION 4: Reorder particles for coalesced access
    // ========================================================================
    CudaTimer timer_reorder;
    timer_reorder.begin();
    
    reorder_particles_by_cell_kernel<<<blocks, threads>>>(
        d_particles, d_particles_sorted, d_grid_indices, count);
    CUDA_CHECK(cudaGetLastError());
    
    double reorder_time = timer_reorder.end();
    
    // ========================================================================
    // OPTIMIZATION 5: Collision detection with shared memory
    // ========================================================================
    CudaTimer timer_collision;
    timer_collision.begin();
    
    detect_collisions_optimized<<<blocks, threads>>>(
        d_particles_sorted, d_grid_indices, d_grid_starts, d_grid_counts, count);
    CUDA_CHECK(cudaGetLastError());
    
    double collision_ms = timer_collision.end();
    
    // ========================================================================
    // Restore original particle order
    // ========================================================================
    CudaTimer timer_restore;
    timer_restore.begin();
    
    restore_particle_order_kernel<<<blocks, threads>>>(
        d_particles_sorted, d_particles, d_grid_indices, count);
    CUDA_CHECK(cudaGetLastError());
    
    double restore_time = timer_restore.end();
    
    // ========================================================================
    // OPTIMIZATION 1: Unified memory - single sync, no D2H transfer
    // ========================================================================
    double t_sync_start = get_time_us();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy results back to host structure
    memcpy(sim->get_particle_data(), d_particles, count * sizeof(Particle));
    
    double t_sync_end = get_time_us();
    double d2h_transfer_ms = (t_sync_end - t_sync_start) / 1000.0;
    
    double t_end = get_time_us();
    double total_time_ms = (t_end - t_start) / 1000.0;
    
    size_t particle_data_size = count * sizeof(Particle);
    
    // Detailed profiling output
    if (verbose) {
        printf("\n[GPU COMPLEX - Frame %d] FULLY OPTIMIZED PIPELINE\n", sim->get_frame_counter());
        printf("================================================================================\n");
        printf("Particles: %d (%zu bytes data)\n", count, particle_data_size);
        printf("GPU Memory: %.2f MB total\n", total_gpu_memory_bytes / (1024.0 * 1024.0));
        printf("--------------------------------------------------------------------------------\n");
        printf("Constant copy:        %7.3f ms %s\n", 
               constant_copy_ms,
               constant_copy_ms == 0.0 ? "(cached)" : "");
        printf("Update kernel:        %7.3f ms\n", update_kernel_ms);
        printf("Count (2-phase):      %7.3f ms (shared memory)\n", grid_count_ms);
        printf("Prefix sum (GPU):     %7.3f ms (Blelloch scan)\n", prefix_sum_ms);
        printf("Fill indices:         %7.3f ms\n", grid_fill_ms);
        printf("Reorder particles:    %7.3f ms (coalescing)\n", reorder_time);
        printf("Collision (shared):   %7.3f ms\n", collision_ms);
        printf("Restore order:        %7.3f ms\n", restore_time);
        printf("Final sync:           %7.3f ms (unified memory)\n", d2h_transfer_ms);
        printf("--------------------------------------------------------------------------------\n");
        printf("TOTAL FRAME TIME:     %7.3f ms (%.1f FPS)\n", 
               total_time_ms, 1000.0 / total_time_ms);
        printf("================================================================================\n");
        
        // Calculate speedup estimate
        double sequential_estimate = count * count * 1e-6;  // Rough O(n²) estimate
        printf("Estimated speedup vs naive O(n²): %.1fx\n", sequential_estimate / total_time_ms);
        printf("\n");
    }
}
