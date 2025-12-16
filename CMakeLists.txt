cmake_minimum_required(VERSION 3.18)
project(ParticleSimulation LANGUAGES CXX)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        add_compile_options(/O2 /W3)
    else()
        add_compile_options(-O3 -march=native -Wall)
    endif()
endif()

# Platform detection
if(WIN32)
    add_definitions(-DPLATFORM_WINDOWS)
    message(STATUS "Building for Windows Desktop")
elseif(UNIX AND NOT APPLE)
    add_definitions(-DPLATFORM_LINUX)
    if(EXISTS "/etc/nv_tegra_release")
        add_definitions(-DPLATFORM_JETSON)
        message(STATUS "Building for NVIDIA Jetson")
    else()
        message(STATUS "Building for Linux Desktop")
    endif()
endif()

# Source files
set(SOURCES
    Main.cpp
    Simulation.cpp
    Physics.cpp
    Rendering.cpp
    SystemMonitor.cpp
)

# Find SDL2
if(WIN32)
    if(MINGW)
        # MinGW: Use pkg-config (requires SDL2 installed via MSYS2)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(SDL2 REQUIRED sdl2)
        pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)
        
        include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_TTF_INCLUDE_DIRS})
        link_directories(${SDL2_LIBRARY_DIRS} ${SDL2_TTF_LIBRARY_DIRS})
        
        set(SDL2_LIBRARIES ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES})
    else()
        # MSVC: SDL2 in C:/SDL2 and C:/SDL2_ttf
        set(SDL2_INCLUDE_DIR "C:/SDL2/include" "C:/SDL2_ttf/include")
        set(SDL2_LIBRARY_DIR "C:/SDL2/lib/x64" "C:/SDL2_ttf/lib/x64")
        
        include_directories(${SDL2_INCLUDE_DIR})
        link_directories(${SDL2_LIBRARY_DIR})
        
        set(SDL2_LIBRARIES SDL2 SDL2main SDL2_ttf)
    endif()
else()
    # Linux/Jetson: use pkg-config
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(SDL2 REQUIRED sdl2)
    pkg_check_modules(SDL2_TTF REQUIRED SDL2_ttf)
    
    include_directories(${SDL2_INCLUDE_DIRS} ${SDL2_TTF_INCLUDE_DIRS})
    link_directories(${SDL2_LIBRARY_DIRS} ${SDL2_TTF_LIBRARY_DIRS})
    
    set(SDL2_LIBRARIES ${SDL2_LIBRARIES} ${SDL2_TTF_LIBRARIES})
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - Multithreaded mode available")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found - Multithreaded mode will fall back to sequential")
endif()

# MPI support (optional)
option(USE_MPI "Build with MPI support" OFF)
if(USE_MPI)
    find_package(MPI)
    if(MPI_CXX_FOUND)
        message(STATUS "MPI found - MPI mode available")
        include_directories(${MPI_CXX_INCLUDE_DIRS})
        add_definitions(-DUSE_MPI)
        set(MPI_LIBRARIES ${MPI_CXX_LIBRARIES})
    else()
        message(WARNING "MPI requested but not found - MPI mode will fall back to sequential")
    endif()
endif()

# CUDA support (optional)
option(USE_CUDA "Build with CUDA support" ON)
if(USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA found - GPU modes available")
        
        # Detect GPU architecture
        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Default to common architectures
            # sm_86 = RTX 30xx/40xx, Xavier NX
            # sm_75 = Turing (RTX 20xx)
            # sm_72 = Xavier AGX
            # sm_53 = Jetson Nano
            set(CMAKE_CUDA_ARCHITECTURES 86)
            message(STATUS "Using CUDA architecture: sm_86 (RTX 3050 / Xavier NX)")
            message(STATUS "Override with: -DCMAKE_CUDA_ARCHITECTURES=XX")
        endif()
        
        set(CMAKE_CUDA_STANDARD 17)
        add_definitions(-DUSE_CUDA)
        
        # Add CUDA source
        list(APPEND SOURCES PhysicsGPU.cu)
        
        set_source_files_properties(PhysicsGPU.cu PROPERTIES LANGUAGE CUDA)
    else()
        message(WARNING "CUDA requested but not found - GPU modes will fall back to sequential")
    endif()
endif()

# Create executable
add_executable(particle_sim ${SOURCES})

# Link libraries
target_link_libraries(particle_sim ${SDL2_LIBRARIES})

if(OpenMP_CXX_FOUND)
    target_link_libraries(particle_sim OpenMP::OpenMP_CXX)
endif()

if(USE_MPI AND MPI_CXX_FOUND)
    target_link_libraries(particle_sim ${MPI_LIBRARIES})
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(particle_sim cudart)
endif()

# Math library on Unix
if(UNIX)
    target_link_libraries(particle_sim m)
endif()

# Windows subsystem
if(WIN32)
    if(MSVC)
        set_target_properties(particle_sim PROPERTIES
            LINK_FLAGS "/SUBSYSTEM:CONSOLE"
        )
    endif()
endif()

# Installation
install(TARGETS particle_sim DESTINATION bin)

# Print build configuration
message(STATUS "")
message(STATUS "================================")
message(STATUS "Build Configuration Summary")
message(STATUS "================================")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP: YES (Multithreaded mode enabled)")
else()
    message(STATUS "OpenMP: NO (Multithreaded mode will fall back)")
endif()

if(USE_MPI AND MPI_CXX_FOUND)
    message(STATUS "MPI: YES (MPI mode enabled)")
else()
    message(STATUS "MPI: NO (MPI mode will fall back)")
endif()

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    message(STATUS "CUDA: YES (GPU modes enabled)")
    message(STATUS "CUDA Architecture: sm_${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA: NO (GPU modes will fall back)")
endif()

message(STATUS "SDL2: ${SDL2_LIBRARIES}")
message(STATUS "================================")
message(STATUS "")
