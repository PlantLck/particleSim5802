#include "ParticleSimulation.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// ============================================================================
// Platform-Specific System Monitoring
// ============================================================================

#ifdef PLATFORM_JETSON

float SystemMonitor::read_temperature() {
    float max_temp = 0.0f;
    
    // Try multiple thermal zones
    for (int zone = 0; zone < 10; zone++) {
        std::string path = "/sys/devices/virtual/thermal/thermal_zone" + 
                          std::to_string(zone) + "/temp";
        std::ifstream file(path);
        
        if (file.is_open()) {
            int temp_millidegrees;
            file >> temp_millidegrees;
            float temp_celsius = temp_millidegrees / 1000.0f;
            if (temp_celsius > max_temp) {
                max_temp = temp_celsius;
            }
        }
    }
    
    // Try alternative path if no zones found
    if (max_temp == 0.0f) {
        std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
        if (file.is_open()) {
            int temp_millidegrees;
            file >> temp_millidegrees;
            max_temp = temp_millidegrees / 1000.0f;
        }
    }
    
    return max_temp;
}

float SystemMonitor::read_power() {
    float total_power = 0.0f;
    int rail_count = 0;
    
    // Try multiple power rail paths
    const std::vector<std::string> power_paths = {
        "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device0/in_power0_input",
        "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input",
        "/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input"
    };
    
    for (const auto& path : power_paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            int power_milliwatts;
            file >> power_milliwatts;
            total_power += power_milliwatts / 1000.0f;
            rail_count++;
        }
    }
    
    return total_power;
}

#elif defined(PLATFORM_LINUX)

// Generic Linux (desktop)
float SystemMonitor::read_temperature() {
    float max_temp = 0.0f;
    
    // Try reading CPU temperature
    std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
    if (file.is_open()) {
        int temp_millidegrees;
        file >> temp_millidegrees;
        max_temp = temp_millidegrees / 1000.0f;
    }
    
    return max_temp;
}

float SystemMonitor::read_power() {
    // Power monitoring not readily available on generic Linux desktop
    return 0.0f;
}

#elif defined(PLATFORM_WINDOWS)

// Windows - no easy way to read these without admin privileges
float SystemMonitor::read_temperature() {
    return 0.0f;  // Would require WMI or admin privileges
}

float SystemMonitor::read_power() {
    return 0.0f;  // Would require performance counters or admin privileges
}

#else

// Unknown platform
float SystemMonitor::read_temperature() {
    return 0.0f;
}

float SystemMonitor::read_power() {
    return 0.0f;
}

#endif

// ============================================================================
// Update Metrics (Platform-Independent)
// ============================================================================

void SystemMonitor::update_metrics(Simulation& sim) {
    static int update_counter = 0;
    
    // Update every 30 frames to reduce overhead
    update_counter++;
    if (update_counter >= 30) {
        sim.set_temperature(read_temperature());
        sim.set_power(read_power());
        update_counter = 0;
    }
}
