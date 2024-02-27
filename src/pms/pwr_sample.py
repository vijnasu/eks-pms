from termcolor import colored
import sys
import subprocess
import threading
from intelligent_profile_manager import intelligent_apply_profiles
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

class FrequencyConfigurator:
    def __init__(self, cores, type_name):
        self.cores = cores  # Assuming 'cores' is a list of Core instances
        self.type_name = type_name
        self.console = Console()

    def display_current_configurations(self):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column(f"{self.type_name} ID", style="dim", width=12)
        table.add_column("Current Frequency", justify="right")
        table.add_column("Minimum Frequency", justify="right")
        table.add_column("Maximum Frequency", justify="right")

        for core in self.cores:
            if core.online:
                table.add_row(
                    str(core.core_id),
                    f"{core.curr_freq} MHz" if core.curr_freq is not None else "N/A",
                    f"{core.min_freq} MHz" if core.min_freq is not None else "N/A",
                    f"{core.max_freq} MHz" if core.max_freq is not None else "N/A"
                )
            else:
                table.add_row(str(core.core_id), "Offline", "Offline", "Offline")

        self.console.print(f"Current {self.type_name} Configurations:", style="bold magenta")
        self.console.print(table)

    def batch_adjust_configurations(self):
        # Display current configurations before adjustments
        self.display_current_configurations()

        # Ask for new frequencies only once
        new_min_freq = Prompt.ask(f"Enter new minimum frequency (MHz) for all {self.type_name.lower()} cores", default="N/A")
        new_max_freq = Prompt.ask(f"Enter new maximum frequency (MHz) for all {self.type_name.lower()} cores", default="N/A")

        for core in self.cores:
            if core.online:
                core.min_freq = int(new_min_freq) if new_min_freq.isdigit() else core.min_freq
                core.max_freq = int(new_max_freq) if new_max_freq.isdigit() else core.max_freq

        # Display configurations after adjustments
        self.console.print(f"\n[bold cyan]After Adjustments:[/bold cyan]")
        self.display_current_configurations()

class Uncore:
    def __init__(self, uncore_id, min_freq, max_freq, curr_freq):
        self.uncore_id = uncore_id
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.curr_freq = curr_freq

def initialize_uncore_regions(cpus):
    uncore_regions = []
    for cpu in cpus:
        uncore_id = cpu.cpu_id
        min_freq = cpu.uncore_hw_min
        max_freq = cpu.uncore_hw_max
        curr_freq = cpu.uncore_freq

        # Create and add the Uncore object to the list
        uncore_region = Uncore(uncore_id, min_freq, max_freq, curr_freq)
        uncore_regions.append(uncore_region)

    return uncore_regions

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
    
def commit_changes_concurrently(cores):
    threads = []
    for core in cores:
        thread = threading.Thread(target=commit_core_changes, args=(core,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print(colored("Concurrently committed changes for all cores.", "green"))

def intelligent_apply_profiles(cores, default_profile="default"):
    for core in cores:
        # Placeholder for intelligent decision-making, e.g., based on core usage or temperature
        profile = determine_profile_for_core(core)  # You need to define this function
        core.commit(profile)
    print(colored("Intelligently applied profiles based on core metrics.", "green"))

# Global Refresh with Local Consideration
# Refreshing stats globally but with consideration for minimizing performance impact:
    
def staggered_refresh_stats(system, cpus, cores):
    system.refresh_stats()  # Refresh system-wide stats
    for cpu in cpus:
        cpu.refresh_stats()  # Refresh CPU-specific stats
        for core in cpu.core_list:
            core.refresh_stats()  # Refresh core-specific stats, possibly staggered
    print(colored("Staggered refresh of stats completed.", "green"))
    
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
    options = [
        "Adjust Core Configuration",
        "Adjust CPU Uncore Configuration",
        "Commit Changes",
        "Apply Preset Profile",
        "Refresh Stats",
        "Show Object Referencing",
        "Adjust EPP and Show Power Consumption",
        "Configure C-States",
        "Check Configuration Stability",
        "Intelligently Apply Profiles",  # New option for intelligent profile application
        "Exit"
    ]
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
            core_configurator = FrequencyConfigurator(cores, "Core")
            core_configurator.batch_adjust_configurations()
        elif choice == "2":
            uncore_regions = initialize_uncore_regions(cpus)
            uncore_configurator = FrequencyConfigurator(uncore_regions, "Uncore")
            uncore_configurator.batch_adjust_configurations()
        elif choice == "3":
            commit_changes_concurrently(cores)
        elif choice == "4":
            apply_preset_profile(cores, "preset_profile_here")  # Replace "preset_profile_here" with actual profile name
        elif choice == "5":
            refresh_stats(system, cpus, cores)
        elif choice == "6":
            # Assuming we want to show object referencing for a specific core, we might need to select a core or iterate over cores
            for core in cores:
                show_object_referencing(core)
        elif choice == "7":
            for core in cores:
                adjust_epp_and_show_power_consumption(core, "new_epp_value_here")  # Replace "new_epp_value_here" with actual EPP value
        elif choice == "8":
            for core in cores:
                configure_cstates(core, "C-State_Name", True)  # Replace "C-State_Name" with actual C-State name and True/False to enable/disable
        elif choice == "9":
            request_configuration_stability(system)
        elif choice == "10":
            intelligent_apply_profiles(cores)
        elif choice == "11":
            exit_program()
        else:
            print(colored("Invalid choice, please try again.", "red"))

if __name__ == "__main__":
    main()