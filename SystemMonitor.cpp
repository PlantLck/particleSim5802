/*
 * System Monitoring Implementation
 * Platform-specific temperature and power measurement
 */

#include "ParticleSimulation.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>

// ============================================================================
// Platform-Specific Implementations
// ============================================================================

#ifdef PLATFORM_JETSON

float SystemMonitor::read_temperature() {
    float max_temp = 0.0f;
    
    // Scan multiple thermal zones to find highest temperature
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
    
    // Fallback to standard thermal zone if no zones found
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
    
    // Jetson power monitoring via INA3221x sensors
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
        }
    }
    
    return total_power;
}

#elif defined(PLATFORM_LINUX)

float SystemMonitor::read_temperature() {
    float max_temp = 0.0f;
    
    // Generic Linux thermal monitoring
    std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
    if (file.is_open()) {
        int temp_millidegrees;
        file >> temp_millidegrees;
        max_temp = temp_millidegrees / 1000.0f;
    }
    
    return max_temp;
}

float SystemMonitor::read_power() {
    // Desktop Linux power monitoring not implemented
    // Would require RAPL interface or hardware-specific drivers
    return 0.0f;
}

#else

#error "Unsupported platform - only Linux and Jetson are supported"

#endif

// ============================================================================
// Platform-Independent Interface
// ============================================================================

void SystemMonitor::update_metrics(Simulation& sim) {
    static int update_counter = 0;
    
    // Update every 30 frames to minimize performance impact
    update_counter++;
    if (update_counter >= 30) {
        sim.set_temperature(read_temperature());
        sim.set_power(read_power());
        update_counter = 0;
    }
}
