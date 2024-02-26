import pwr
from termcolor import colored
import sys
import subprocess

def check_and_install_pwr():
    try:
        import pwr
        print(colored("pwr module is already installed.", "green"))
    except ImportError:
        print(colored("pwr module not found. Installing...", "yellow"))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/intel/CommsPowerManagement.git#egg=pwr&subdirectory=pwr"])
        print(colored("pwr module installed successfully.", "green"))

def initialize_objects():
    check_and_install_pwr()
    import pwr  # Import here to ensure it's installed
    system, cpus, cores = pwr.get_objects()
    print_objects_info(system, cpus, cores)
    return system, cpus, cores

def print_objects_info(system, cpus, cores):
    print(colored(f"System initialized with {len(cpus)} CPUs and {len(cores)} cores.", "green"))

def adjust_core_configuration(core):
    core.min_freq = core.lowest_freq
    core.max_freq = core.highest_freq
    print(colored(f"Core {core.core_id}: Min freq set to {core.min_freq}, Max freq set to {core.max_freq}", "cyan"))

def adjust_cpu_uncore_configuration(cpu):
    cpu.uncore_min_freq = cpu.uncore_hw_min
    cpu.uncore_max_freq = cpu.uncore_hw_max
    print(colored(f"CPU {cpu.cpu_id}: Uncore Min freq set to {cpu.uncore_min_freq}, Uncore Max freq set to {cpu.uncore_max_freq}", "cyan"))

def commit_core_changes(cores):
    for core in cores:
        core.commit()
        print(colored(f"Changes committed for Core {core.core_id}.", "green"))

def commit_system_changes(system):
    system.commit()
    print(colored("Changes committed for the entire system.", "green"))

def apply_preset_profile(cores, profile):
    for core in cores:
        core.commit(profile)
    print(colored(f"Applied '{profile}' profile to all cores.", "green"))

def refresh_stats(system, cpus, cores):
    system.refresh_all()
    print(colored("All stats refreshed.", "green"))

def show_object_referencing(core):
    print(colored(f"Core {core.core_id} is on CPU {core.cpu.cpu_id} in System with SST-BF enabled: {core.cpu.sys.sst_bf_enabled}", "cyan"))

def adjust_epp_and_show_power_consumption(core, epp_value):
    core.epp = epp_value
    print(colored(f"Core {core.core_id}: EPP set to {core.epp}. Current power consumption: {core.cpu.power_consumption}W", "cyan"))

def configure_cstates(core, cstate, enable):
    core.cstates[cstate] = enable
    core.commit()
    state = "enabled" if enable else "disabled"
    print(colored(f"{cstate} state {state} for Core {core.core_id}.", "green"))

def request_configuration_stability(system):
    if system.request_configuration():
        print(colored("The current configuration is stable.", "green"))
    else:
        print(colored("The current configuration may not be stable.", "red"))

def display_menu():
    print(colored("\nChoose an option:", "yellow"))
    options = ["Adjust Core Configuration", "Adjust CPU Uncore Configuration", "Commit Changes", "Apply Preset Profile", "Refresh Stats", "Show Object Referencing", "Adjust EPP and Show Power Consumption", "Configure C-States", "Check Configuration Stability", "Exit"]
    for i, option in enumerate(options, 1):
        print(colored(f"{i}. {option}", "yellow"))

def exit_program():
    print(colored("\nExiting program.", "red"))
    sys.exit()

def main():
    system, cpus, cores = initialize_objects()
    while True:
        display_menu()
        choice = input(colored("Enter your choice: ", "green"))
        if choice == "1":
            for core in cores:
                adjust_core_configuration(core)
        elif choice == "2":
            for cpu in cpus:
                adjust_cpu_uncore_configuration(cpu)
        # Add more elif blocks for other options
        elif choice == "10":
            exit_program()
        else:
            print(colored("Invalid choice, please try again.", "red"))

if __name__ == "__main__":
    main()
