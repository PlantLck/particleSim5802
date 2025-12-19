/*
 * System Monitoring Implementation
 * Platform-specific temperature and power measurement
 */

#include "ParticleSimulation.hpp"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <dirent.h>

#ifdef PLATFORM_JETSON

float SystemMonitor::read_temperature() {
    float max_temp = 0.0f;
    
    for (int zone = 0; zone < 10; zone++) {
        std::string path = "/sys/devices/virtual/thermal/thermal_zone" + 
                          std::to_string(zone) + "/temp";
        std::ifstream file(path);
        
        if (file.is_open()) {
            int temp_millidegrees;
            if (file >> temp_millidegrees) {
                float temp_celsius = temp_millidegrees / 1000.0f;
                if (temp_celsius > max_temp && temp_celsius < 150.0f) {
                    max_temp = temp_celsius;
                }
            }
        }
    }
    
    if (max_temp == 0.0f) {
        std::ifstream file("/sys/class/thermal/thermal_zone0/temp");
        if (file.is_open()) {
            int temp_millidegrees;
            if (file >> temp_millidegrees) {
                max_temp = temp_millidegrees / 1000.0f;
            }
        }
    }
    
    return max_temp;
}

float SystemMonitor::read_power() {
    float total_power = 0.0f;
    
    const std::vector<std::string> power_paths = {
        "/sys/bus/i2c/drivers/ina3221x/0-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/0-0041/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0040/iio:device0/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221x/1-0041/iio:device0/in_power0_input",
        "/sys/devices/3160000.i2c/i2c-0/0-0040/iio_device/in_power0_input",
        "/sys/devices/3160000.i2c/i2c-0/0-0041/iio_device/in_power0_input",
        "/sys/bus/i2c/drivers/ina3221/0-0040/hwmon/hwmon0/power1_input",
        "/sys/bus/i2c/drivers/ina3221/0-0041/hwmon/hwmon0/power1_input"
    };
    
    for (const auto& path : power_paths) {
        std::ifstream file(path);
        if (file.is_open()) {
            int power_milliwatts;
            if (file >> power_milliwatts) {
                total_power += power_milliwatts / 1000.0f;
            }
        }
    }
    
    return total_power;
}

#elif defined(PLATFORM_LINUX)

float SystemMonitor::read_temperature() {
    float max_temp = 0.0f;
    
    for (int zone = 0; zone < 10; zone++) {
        std::string path = "/sys/class/thermal/thermal_zone" + 
                          std::to_string(zone) + "/temp";
        std::ifstream file(path);
        
        if (file.is_open()) {
            int temp_millidegrees;
            if (file >> temp_millidegrees) {
                float temp_celsius = temp_millidegrees / 1000.0f;
                if (temp_celsius > max_temp && temp_celsius < 150.0f) {
                    max_temp = temp_celsius;
                }
            }
        }
    }
    
    if (max_temp == 0.0f) {
        const std::vector<std::string> hwmon_paths = {
            "/sys/class/hwmon/hwmon0/temp1_input",
            "/sys/class/hwmon/hwmon1/temp1_input",
            "/sys/class/hwmon/hwmon2/temp1_input",
            "/sys/class/hwmon/hwmon3/temp1_input"
        };
        
        for (const auto& path : hwmon_paths) {
            std::ifstream file(path);
            if (file.is_open()) {
                int temp_millidegrees;
                if (file >> temp_millidegrees) {
                    float temp_celsius = temp_millidegrees / 1000.0f;
                    if (temp_celsius > max_temp && temp_celsius < 150.0f) {
                        max_temp = temp_celsius;
                    }
                }
            }
        }
    }
    
    return max_temp;
}

float SystemMonitor::read_power() {
    float total_power = 0.0f;
    
    std::ifstream energy_file("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    if (energy_file.is_open()) {
        static unsigned long long last_energy = 0;
        static double last_time = 0.0;
        
        unsigned long long current_energy;
        if (energy_file >> current_energy) {
            double current_time = Utils::get_time_ms() / 1000.0;
            
            if (last_energy > 0 && last_time > 0.0) {
                double time_diff = current_time - last_time;
                if (time_diff > 0.0) {
                    unsigned long long energy_diff = current_energy - last_energy;
                    total_power = (energy_diff / 1000000.0f) / time_diff;
                }
            }
            
            last_energy = current_energy;
            last_time = current_time;
        }
    }
    
    return total_power;
}

#else

float SystemMonitor::read_temperature() {
    return 0.0f;
}

float SystemMonitor::read_power() {
    return 0.0f;
}

#endif

void SystemMonitor::update_metrics(Simulation& sim) {
    static int update_counter = 0;
    
    update_counter++;
    if (update_counter >= 15) {
        float temp = read_temperature();
        float power = read_power();
        
        sim.set_temperature(temp);
        sim.set_power(power);
        
        update_counter = 0;
    }
}
