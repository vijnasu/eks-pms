#!/usr/bin/env python3
import os
import sys
import argparse

# Define the paths to the sysfs entries for CPU frequency and power management
cpu_base_path = "/sys/devices/system/cpu"

# get_governers function
def get_governors():
    """Retrieves the available CPU governors."""
    gov_path = f"{cpu_base_path}/cpu0/cpufreq/scaling_available_governors"
    with open(gov_path, 'r') as f:
        return f.read().strip().split()
    
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

# get_min_frequency
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

# get_governors function
def validate_governor(governor):
    """Validates if the provided governor is supported."""
    supported_governors = get_governors()  # Assuming get_governors() returns a list of available governors
    if governor not in supported_governors:
        raise ValueError(f"Unsupported governor '{governor}'. Supported governors are: {supported_governors}")

# validate_frequency function
def validate_frequency(core, frequency, min_or_max):
    """Validates if the provided frequency is within the supported range for the core."""
    if min_or_max == "min":
        max_freq = get_max_frequency(core)
        if frequency > max_freq:
            raise ValueError(f"Minimum frequency {frequency} MHz cannot be higher than the current maximum frequency {max_freq} MHz for core {core}")
    elif min_or_max == "max":
        min_freq = get_min_frequency(core)
        if frequency < min_freq:
            raise ValueError(f"Maximum frequency {frequency} MHz cannot be lower than the current minimum frequency {min_freq} MHz for core {core}")

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
        # Validate core range
        total_cores = os.cpu_count()
        if any(core < 0 or core >= total_cores for core in core_range):
            raise ValueError(f"Core range is out of bounds. System has {total_cores} cores.")
    else:
        core_range = range(os.cpu_count())

    # Display current settings
    print("Before changes:")
    for core in core_range:
        print(f"Core {core}: Governor: {get_current_governor(core)}, Frequency: {get_current_frequency(core)} MHz, Min: {get_min_frequency(core)} MHz, Max: {get_max_frequency(core)} MHz")
    print(f"Turbo Boost: {get_turbo_status()}")

    # Apply changes
    if args.governor:
        validate_governor(args.governor)  # Validate governor
        set_governor(core_range, args.governor)
    if args.max:
        for core in core_range:
            validate_frequency(core, args.max, "max")  # Validate max frequency
        set_frequency(core_range, max_freq=args.max)
    if args.min:
        for core in core_range:
            validate_frequency(core, args.min, "min")  # Validate min frequency
        set_frequency(core_range, min_freq=args.min)
    if args.set:
        set_frequency(core_range, set_freq=args.set)  # Direct frequency setting might not require validation
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