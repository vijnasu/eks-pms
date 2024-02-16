#!/usr/bin/env python3
import os
import sys
import argparse

# Define the paths to the sysfs entries for CPU frequency and power management
cpu_base_path = "/sys/devices/system/cpu"

# get_current_governor function
def get_current_governor(core):
    """Retrieves the current CPU governor for a specified core."""
    gov_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_governor"
    with open(gov_path, 'r') as f:
        return f.read().strip()
    
# get_current_frequency function
def get_current_frequency(core):
    """Retrieves the current CPU frequency for a specified core."""
    freq_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_cur_freq"
    with open(freq_path, 'r') as f:
        return int(f.read().strip()) // 1000  # Convert from kHz to MHz

# get_min_frequency and get_max_frequency functions
def get_min_frequency(core):
    """Retrieves the current CPU frequency for a specified core."""
    freq_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_min_freq"
    with open(freq_path, 'r') as f:
        return int(f.read().strip()) // 1000  # Convert from kHz to MHz

# get_max_frequency function    
def get_max_frequency(core):
    """Retrieves the current CPU frequency for a specified core."""
    freq_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_max_freq"
    with open(freq_path, 'r') as f:
        return int(f.read().strip()) // 1000  # Convert from kHz to MHz

# get_turbo_status function    
def get_turbo_status():
    """Retrieves the current Turbo Boost status."""
    turbo_path = "/sys/devices/system/cpu/intel_pstate/no_turbo"
    with open(turbo_path, 'r') as f:
        return 'Disabled' if f.read().strip() == '1' else 'Enabled'

# set_governor function
def set_governor(core_range, governor):
    """Sets the CPU governor for the specified core range."""
    for core in core_range:
        gov_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_governor"
        with open(gov_path, 'w') as f:
            f.write(governor)

# set_frequency function
def set_frequency(core_range, min_freq=None, max_freq=None, set_freq=None):
    """Sets the CPU frequency limits for the specified core range."""
    for core in core_range:
        if min_freq:
            min_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_min_freq"
            with open(min_path, 'w') as f:
                f.write(str(min_freq))
        if max_freq:
            max_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_max_freq"
            with open(max_path, 'w') as f:
                f.write(str(max_freq))
        if set_freq:
            set_path = f"{cpu_base_path}/cpu{core}/cpufreq/scaling_setspeed"
            with open(set_path, 'w') as f:
                f.write(str(set_freq))

# enable_turbo function
def enable_turbo(enable=True):
    """Enables or disables Turbo Boost."""
    turbo_path = "/sys/devices/system/cpu/intel_pstate/no_turbo"
    with open(turbo_path, 'w') as f:
        f.write('0' if enable else '1')

# parse_core_range function
def parse_core_range(core_range):
    """Parses a core range string and returns a list of core numbers."""
    cores = []
    for part in core_range.split(','):
        if '-' in part:
            start, end = map(int, part.split('-'))
            cores.extend(range(start, end + 1))
        else:
            cores.append(int(part))
    return cores

# main function
def main():
    parser = argparse.ArgumentParser(description="CPU Power Management Configuration Script")
    parser.add_argument('-g', '--governor', help="Set CPU governor")
    parser.add_argument('-r', '--range', help="Range of cores to affect, e.g., 0-3,5")
    parser.add_argument('-M', '--max', type=int, help="Set core maximum frequency")
    parser.add_argument('-m', '--min', type=int, help="Set core minimum frequency")
    parser.add_argument('-s', '--set', type=int, help="Set core frequency")
    parser.add_argument('-T', '--turbo', action='store_true', help="Enable Turbo Boost")
    parser.add_argument('-t', '--no-turbo', action='store_true', help="Disable Turbo Boost")

    args = parser.parse_args()

    if args.range:
        core_range = parse_core_range(args.range)
    else:
        core_range = range(os.cpu_count())    
    
    # Display current, min, max settings
    print("Before changes:")
    for core in core_range:
        print(f"Core {core}: Governor: {get_current_governor(core)}, Frequency: {get_current_frequency(core)} MHz, Min: {get_min_frequency(core)} MHz, Max: {get_max_frequency(core)} MHz")
    print(f"Turbo Boost: {get_turbo_status()}")
    
    # Apply changes
    if args.governor:
        set_governor(core_range, args.governor)
    if args.max or args.min or args.set:
        set_frequency(core_range, min_freq=args.min, max_freq=args.max, set_freq=args.set)
    if args.turbo:
        enable_turbo(enable=True)
    if args.no_turbo:
        enable_turbo(enable=False)

    # Display new settings
    print("\nAfter changes:")
    for core in core_range:
        print(f"Core {core}: Governor: {get_current_governor(core)}, Frequency: {get_current_frequency(core)} MHz, Min: {get_min_frequency(core)} MHz, Max: {get_max_frequency(core)} MHz")
    print(f"Turbo Boost: {get_turbo_status()}")
            
if __name__ == "__main__":
    main()