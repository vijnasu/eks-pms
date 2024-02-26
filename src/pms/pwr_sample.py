import pwr

# Get system, cpu and core objects
system, cpus, cores = pwr.get_objects()

# Print system attributes
print("System SST-BF Enabled: ", system.sst_bf_enabled)
print("System SST-BF Configured: ", system.sst_bf_configured)

# Print CPU attributes
for cpu in cpus:
    print("CPU ID: ", cpu.cpu_id)
    print("CPU Base Frequency: ", cpu.base_freq)
    print("CPU Power Consumption: ", cpu.power_consumption)

# Print Core attributes
for core in cores:
    print("Core ID: ", core.core_id)
    print("Core Base Frequency: ", core.base_freq)
    print("Core Current Frequency: ", core.curr_freq)

# Modify core frequency
for core in cores:
    core.min_freq = core.lowest_freq
    core.max_freq = core.highest_freq
    core.commit()

# Refresh CPU stats
for cpu in cpus:
    cpu.refresh_stats()
    print("Updated CPU Power Consumption: ", cpu.power_consumption)
