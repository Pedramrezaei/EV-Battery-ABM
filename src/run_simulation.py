import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import to_rgba
import numpy as np
import time
import matplotlib

root_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(root_dir))

from src.models.battery_circularity_model import BatteryCircularityModel
from src.agents.eVOwner import EVOwner
from src.agents.car_manufacturer import CarManufacturer
from src.agents.recycling_facility import RecyclingFacility
from src.agents.battery_refurbisher import BatteryRefurbisher
from src.agents.battery import BatteryStatus
from src.utils.constants import (
    DEFAULT_NUM_OWNERS,
    DEFAULT_NUM_MANUFACTURERS,
    DEFAULT_NUM_RECYCLERS,
    DEFAULT_NUM_REFURBISHERS,
    DEFAULT_SIMULATION_STEPS
)

try:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from generate_figures import scenarios
    print("Successfully imported scenarios from generate_figures.py")
except ImportError as e:
    print(f"Error importing scenarios: {e}")
    scenarios = {
        "Baseline": {
            "technical_capability": 0.7,
            "refurbisher_capacity": 2,
            "recycling_commitment": 0.75,
            "second_life_threshold": 0.6
        },
        "Enhanced Refurbishment": {
            "technical_capability": 0.9,
            "refurbisher_capacity": 8,
            "recycling_commitment": 0.75,
            "second_life_threshold": 0.6
        },
        "Policy-Driven Recycling": {
            "technical_capability": 0.7,
            "refurbisher_capacity": 3,
            "recycling_commitment": 0.95,
            "second_life_threshold": 0.6
        },
        "Lower Second-Life Threshold": {
            "technical_capability": 0.7,
            "refurbisher_capacity": 2,
            "recycling_commitment": 0.75,
            "second_life_threshold": 0.4
        },
        "Combined Approach": {
            "technical_capability": 0.9,
            "refurbisher_capacity": 8,
            "recycling_commitment": 0.95,
            "second_life_threshold": 0.4
        }
    }

figures_dir = Path(root_dir) / "figures"
figures_dir.mkdir(exist_ok=True)

def run_simulation_for_scenario(scenario_name, params, steps=500, 
                  num_owners=DEFAULT_NUM_OWNERS, 
                  num_manufacturers=DEFAULT_NUM_MANUFACTURERS, 
                  num_recyclers=DEFAULT_NUM_RECYCLERS, 
                  num_refurbishers=DEFAULT_NUM_REFURBISHERS,
                  debug_output=False,
                  save_figures=True,
                  update_interval=10):
    """Run the simulation for a specific scenario and display/save results.
    
    Args:
        scenario_name (str): Name of the scenario being run
        params (dict): Parameters for this scenario
        steps (int): Number of simulation steps to run (default 120 months = 10 years)
        num_owners (int): Number of EV owners
        num_manufacturers (int): Number of car manufacturers
        num_recyclers (int): Number of recycling facilities
        num_refurbishers (int): Number of battery refurbishers
        debug_output (bool): Whether to print detailed debug information
        save_figures (bool): Whether to save figures to files
        update_interval (int): How often to update the display (every N steps)
    """
    # debug mode
    import builtins
    original_print = builtins.print
    
    def filtered_print(*args, **kwargs):
        if debug_output:
            original_print(*args, **kwargs)
    
    if not debug_output:
        builtins.print = filtered_print
    
    try:
        print(f"Starting simulation for scenario '{scenario_name}' with {steps} steps ({steps/12:.1f} years)")
        print(f"Number of owners: {num_owners}")
        print(f"Number of manufacturers: {num_manufacturers}")
        print(f"Number of recyclers: {num_recyclers}")
        print(f"Number of refurbishers: {num_refurbishers}")
        print(f"Scenario params: {params}")

        model = BatteryCircularityModel(
            num_owners=num_owners,
            num_manufacturers=num_manufacturers,
            num_recyclers=num_recyclers,
            num_refurbishers=num_refurbishers
        )

        for manufacturer in model.manufacturers:
            manufacturer.recycling_commitment = params["recycling_commitment"]

        for refurbisher in model.refurbishers:
            refurbisher.technical_capability = params["technical_capability"]
            refurbisher.capacity = params["refurbisher_capacity"]
        
        # set up figure
        fig = plt.figure(figsize=(15, 20))  # Taller figure to accommodate 5 plots
        fig.suptitle(f'EV Battery Circularity Simulation: {scenario_name}', fontsize=16)
        
        # grid visualization
        grid_ax = plt.subplot2grid((5, 3), (0, 0), rowspan=1, colspan=1)
        grid_ax.set_title('Agent Locations')
        grid_ax.set_xticks([])
        grid_ax.set_yticks([])
        
        # battery status
        status_ax = plt.subplot2grid((5, 3), (0, 1), rowspan=1, colspan=2)
        status_ax.set_title('Battery Status Over Time')
        status_ax.set_xlabel('Step')
        status_ax.set_ylabel('Number of Batteries')
        
        # circularity rates
        rates_ax = plt.subplot2grid((5, 3), (1, 0), rowspan=1, colspan=1)
        rates_ax.set_title('Circularity Rates')
        rates_ax.set_xlabel('Step')
        rates_ax.set_ylabel('Rate')
        
        # materials recovered
        materials_ax = plt.subplot2grid((5, 3), (1, 1), rowspan=1, colspan=1)
        materials_ax.set_title('Materials Recovered')
        materials_ax.set_xlabel('Material')
        materials_ax.set_ylabel('Amount (kg)')
        
        # battery age distribution
        age_ax = plt.subplot2grid((5, 3), (1, 2), rowspan=1, colspan=1)
        age_ax.set_title('Battery Age Distribution')
        age_ax.set_xlabel('Age (months)')
        age_ax.set_ylabel('Count')
        
        # facility processing
        facility_ax = plt.subplot2grid((5, 3), (2, 0), rowspan=1, colspan=1)
        facility_ax.set_title('Facility Processing')
        facility_ax.set_xlabel('Step')
        facility_ax.set_ylabel('Count')
        
        # facility inventory levels
        inventory_ax = plt.subplot2grid((5, 3), (2, 1), rowspan=1, colspan=1)
        inventory_ax.set_title('Facility Inventory Levels')
        inventory_ax.set_xlabel('Step')
        inventory_ax.set_ylabel('Count')
        
        # battery flow analysis
        flow_ax = plt.subplot2grid((5, 3), (2, 2), rowspan=1, colspan=1)
        flow_ax.set_title('Battery Flow Analysis')
        flow_ax.set_xlabel('Category')
        flow_ax.set_ylabel('Count')
        
        storage_ax = plt.subplot2grid((5, 3), (3, 0), rowspan=1, colspan=3)
        storage_ax.set_title('Grid Storage Capacity')
        storage_ax.set_xlabel('Step')
        storage_ax.set_ylabel('Capacity (kWh)')
        
        util_ax = plt.subplot2grid((5, 3), (4, 0), rowspan=1, colspan=3)
        util_ax.set_title('Facility Utilization Rates')
        util_ax.set_xlabel('Step')
        util_ax.set_ylabel('Utilization Rate')
        
        # data storage for original plots
        steps_data = []
        active_batteries = []
        end_of_life = []
        recycled = []
        refurbished = []
        recycling_rates = []
        second_life_rates = []
        
        # data storage for facility metrics
        recycler_processed = []
        refurbisher_processed = []
        recycler_inventory = []
        refurbisher_inventory = []
        total_materials_recovered = []
        grid_storage_data = []
        recycler_utilization = []
        refurbisher_utilization = []
        
        # create safe scenario name for filenames
        safe_scenario_name = scenario_name.lower().replace(' ', '_')
        

        for i in range(steps):
            if i % 10 == 0 or i == steps-1:
                print(f"Running step {i+1}/{steps} for scenario '{scenario_name}'")
            
            model.step()

            steps_data.append(i)
            active_batteries.append(model.count_batteries_by_status(BatteryStatus.IN_USE))
            end_of_life.append(model.count_batteries_by_status(BatteryStatus.END_OF_LIFE))
            recycled.append(model.count_batteries_by_status(BatteryStatus.RECYCLED))
            refurbished.append(model.count_batteries_by_status(BatteryStatus.REFURBISHED))
            recycling_rates.append(model.calculate_recycling_rate())
            second_life_rates.append(model.calculate_second_life_rate())

            recycling_stats = model.get_recycling_statistics()
            refurbishment_stats = model.get_refurbishment_statistics()
            
            current_recycler_processed = recycling_stats['total_processed']
            current_refurbisher_processed = refurbishment_stats['total_refurbished']
            recycler_processed.append(current_recycler_processed)
            refurbisher_processed.append(current_refurbisher_processed)
            
            # track inventory levels
            current_recycler_inventory = recycling_stats['queue_length']
            current_refurbisher_inventory = refurbishment_stats['queue_length']
            recycler_inventory.append(current_recycler_inventory)
            refurbisher_inventory.append(current_refurbisher_inventory)
            
            # track materials and grid storage
            materials = recycling_stats['materials_by_type']
            total_materials_recovered.append(sum(materials.values()))
            grid_storage_data.append(model.calculate_total_grid_storage())
            
            # track utilization rates
            r_util = current_recycler_inventory / (num_recyclers * 5) if num_recyclers > 0 else 0
            rf_util = current_refurbisher_inventory / (num_refurbishers * 3) if num_refurbishers > 0 else 0
            recycler_utilization.append(min(1.0, r_util))
            refurbisher_utilization.append(min(1.0, rf_util))
            
            # only update the display every update_interval steps to reduce overhead
            if i % update_interval != 0 and i != steps-1:
                continue
            
            # clear all plots for updating
            grid_ax.clear()
            status_ax.clear()
            rates_ax.clear()
            materials_ax.clear()
            age_ax.clear()
            facility_ax.clear()
            inventory_ax.clear()
            flow_ax.clear()
            storage_ax.clear()
            util_ax.clear()
            
            # reset titles
            grid_ax.set_title('Agent Locations')
            status_ax.set_title('Battery Status Over Time')
            rates_ax.set_title('Circularity Rates')
            materials_ax.set_title('Materials Recovered')
            age_ax.set_title('Battery Age Distribution')
            facility_ax.set_title('Facility Processing')
            inventory_ax.set_title('Facility Inventory Levels')
            flow_ax.set_title('Battery Flow Analysis')
            storage_ax.set_title('Grid Storage Capacity')
            util_ax.set_title('Facility Utilization Rates')
            
            # plot grid
            plot_grid(model, grid_ax)
            
            # plot battery status
            if i > 0:
                status_ax.stackplot(
                    steps_data,
                    active_batteries,
                    end_of_life,
                    recycled,
                    refurbished,
                    labels=['Active', 'End of Life', 'Recycled', 'Refurbished'],
                    colors=['blue', 'red', 'green', 'magenta'],
                    alpha=0.7
                )
                status_ax.legend()
                status_ax.set_xlabel('Step (months)')
                status_ax.set_ylabel('Number of Batteries')
                
            # plot circularity rates
            rates_ax.plot(steps_data, recycling_rates, 'g-', label='Recycling Rate')
            rates_ax.plot(steps_data, second_life_rates, 'm-', label='Second Life Rate')
            rates_ax.legend()
            rates_ax.set_xlabel('Step')
            rates_ax.set_ylabel('Rate')
            
            # plot materials recovered
            materials = recycling_stats['materials_by_type']
            materials_ax.bar(materials.keys(), materials.values())
            materials_ax.set_xlabel('Material')
            materials_ax.set_ylabel('Amount (kg)')
            plt.setp(materials_ax.get_xticklabels(), rotation=45)
            
            # plot battery age distribution
            ages = [agent.battery.age for agent in model.schedule.agents 
                    if isinstance(agent, EVOwner) and agent.battery]
            if ages:
                age_ax.hist(ages, bins=10, alpha=0.7)
                age_ax.set_xlabel('Age (months)')
                age_ax.set_ylabel('Count')
            
            # plot facility processing metrics
            facility_ax.plot(steps_data, recycler_processed, 'r-', label='Recycled')
            facility_ax.plot(steps_data, refurbisher_processed, 'g-', label='Refurbished')
            facility_ax.legend()
            facility_ax.set_xlabel('Step')
            facility_ax.set_ylabel('Cumulative Count')
            
            # plot facility inventory levels
            inventory_ax.plot(steps_data, recycler_inventory, 'r-', label='Recycler Queue')
            inventory_ax.plot(steps_data, refurbisher_inventory, 'g-', label='Refurbisher Queue')
            inventory_ax.legend()
            inventory_ax.set_xlabel('Step')
            inventory_ax.set_ylabel('Current Count')
            
            # plot battery flow analysis
            if i > 0:
                battery_counts = model.get_all_battery_counts()
                
                categories = list(battery_counts.keys())
                values = list(battery_counts.values())
                
                colors = ['blue', 'red', 'orange', 'purple', 'green', 'magenta']
                bars = flow_ax.bar(categories, values, color=colors)
                
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        flow_ax.text(
                            bar.get_x() + bar.get_width()/2.,
                            height + 0.1,
                            f'{int(height)}',
                            ha='center', 
                            va='bottom',
                            fontweight='bold',
                            fontsize=9
                        )
                
                flow_ax.set_xticklabels(categories, rotation=45, ha='right')
                flow_ax.set_ylabel('Number of Batteries')
                flow_ax.set_ylim(bottom=0)
                total_batteries = sum(values)
                flow_ax.set_title(f'Battery Status Distribution (Total: {total_batteries})')
            
            # plot grid storage capacity
            storage_ax.plot(steps_data, grid_storage_data, 'b-', label='Grid Storage')
            storage_ax.legend()
            storage_ax.set_xlabel('Step')
            storage_ax.set_ylabel('Capacity (kWh)')
            
            # plot utilization rates
            util_ax.plot(steps_data, recycler_utilization, 'r-', label='Recycler Utilization')
            util_ax.plot(steps_data, refurbisher_utilization, 'g-', label='Refurbisher Utilization')
            util_ax.legend()
            util_ax.set_xlabel('Step')
            util_ax.set_ylabel('Utilization Rate')
            util_ax.set_ylim(0, 1)
            
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.subplots_adjust(hspace=0.4)
            plt.pause(0.001)

        if save_figures:

            plt.savefig(f"figures/{safe_scenario_name}_complete.png", dpi=300, bbox_inches='tight')

            # BSD
            plt.figure(figsize=(10, 6))
            plt.stackplot(
                steps_data,
                active_batteries,
                end_of_life,
                recycled,
                refurbished,
                labels=['Active', 'End of Life', 'Recycled', 'Refurbished'],
                colors=['blue', 'red', 'green', 'magenta'],
                alpha=0.7
            )
            plt.title(f'Battery Status Distribution - {scenario_name}')
            plt.xlabel('Step (months)')
            plt.ylabel('Number of Batteries')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"figures/{safe_scenario_name}_battery_status.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Circularity Rates
            plt.figure(figsize=(10, 6))
            plt.plot(steps_data, recycling_rates, 'g-', label='Recycling Rate')
            plt.plot(steps_data, second_life_rates, 'm-', label='Second Life Rate')
            plt.title(f'Circularity Rates - {scenario_name}')
            plt.xlabel('Step (months)')
            plt.ylabel('Rate')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"figures/{safe_scenario_name}_circularity_rates.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Facility Processing
            plt.figure(figsize=(10, 6))
            plt.plot(steps_data, recycler_processed, 'r-', label='Recycled')
            plt.plot(steps_data, refurbisher_processed, 'g-', label='Refurbished')
            plt.title(f'Facility Processing - {scenario_name}')
            plt.xlabel('Step (months)')
            plt.ylabel('Cumulative Count')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"figures/{safe_scenario_name}_facility_processing.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Facility Utilization
            plt.figure(figsize=(10, 6))
            plt.plot(steps_data, recycler_utilization, 'r-', label='Recycler Utilization')
            plt.plot(steps_data, refurbisher_utilization, 'g-', label='Refurbisher Utilization')
            plt.title(f'Facility Utilization - {scenario_name}')
            plt.xlabel('Step (months)')
            plt.ylabel('Utilization Rate')
            plt.ylim(0, 1)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"figures/{safe_scenario_name}_facility_utilization.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Grid Storage Capacity
            plt.figure(figsize=(10, 6))
            plt.plot(steps_data, grid_storage_data, 'b-')
            plt.title(f'Grid Storage Capacity - {scenario_name}')
            plt.xlabel('Step (months)')
            plt.ylabel('Capacity (kWh)')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"figures/{safe_scenario_name}_grid_storage.png", dpi=300, bbox_inches='tight')
            plt.close()

        builtins.print = original_print

        recycling_stats = model.get_recycling_statistics()
        refurbishment_stats = model.get_refurbishment_statistics()
        battery_counts = model.get_all_battery_counts()
        
        print("\n=== SIMULATION SUMMARY FOR SCENARIO: " + scenario_name + " ===")
        print(f"Steps run: {steps}")
        print(f"Battery distribution:")
        for status, count in battery_counts.items():
            print(f"  {status}: {count}")
        print(f"Recycling rate: {model.calculate_recycling_rate():.2f}")
        print(f"Second life rate: {model.calculate_second_life_rate():.2f}")
        print(f"Total batteries recycled: {recycling_stats['total_processed']}")
        print(f"Total batteries refurbished: {refurbishment_stats['total_refurbished']}")
        print("==========================\n")
        
        return {
            'scenario_name': scenario_name,
            'active_batteries': active_batteries[-1],
            'end_of_life': end_of_life[-1],
            'recycled': recycler_processed[-1],
            'refurbished': refurbisher_processed[-1],
            'recycling_rate': recycling_rates[-1],
            'second_life_rate': second_life_rates[-1],
            'total_materials_recovered': total_materials_recovered[-1],
            'grid_storage': grid_storage_data[-1],
            'facility_metrics': {
                'recycler_capacity_utilization': recycler_utilization[-1],
                'refurbisher_capacity_utilization': refurbisher_utilization[-1],
            }
        }
    finally:
        builtins.print = original_print

def plot_grid(model, ax):
    """Plot the grid with agents."""
    # Create a grid
    grid = np.zeros((model.grid.width, model.grid.height, 4))
    
    # plot each agent
    for cell_content, (x, y) in model.grid.coord_iter():
        if not cell_content:
            continue
            
        # Handle multiple agents in a cell
        for agent in cell_content:
            if isinstance(agent, EVOwner):
                color = to_rgba('blue', 0.8)

                if agent.battery:

                    health = agent.battery.health
                    if health > 0.8:
                        color = to_rgba('blue', 0.9)  # Good health
                    elif health > 0.6:
                        color = to_rgba('cyan', 0.8)  # Medium health
                    else:
                        color = to_rgba('lightblue', 0.7)  # Poor health
            elif isinstance(agent, CarManufacturer):
                color = to_rgba('green', 0.8)
            elif isinstance(agent, RecyclingFacility):
                color = to_rgba('red', 0.8)
            elif isinstance(agent, BatteryRefurbisher):
                color = to_rgba('purple', 0.8)
            else:
                continue
                
            grid[x][y] = color
    
    ax.imshow(grid.transpose(1, 0, 2))
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add legend
    ax.plot([], [], 'bs', markersize=10, label='EV Owners')
    ax.plot([], [], 'gs', markersize=10, label='Manufacturers')
    ax.plot([], [], 'rs', markersize=10, label='Recyclers')
    ax.plot([], [], 'ms', markersize=10, label='Refurbishers')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

def run_all_scenarios(steps=500, save_figures=True, generate_summary=True):
    """Run all scenarios and save figures for each one."""
    summary_results = {}
    total_scenarios = len(scenarios)
    
    print(f"\nPreparing to run {total_scenarios} scenarios with {steps} steps each...")
    
    for i, (scenario_name, params) in enumerate(scenarios.items(), 1):
        print(f"\n{'='*50}")
        print(f"Running scenario {i}/{total_scenarios}: {scenario_name}")
        print(f"{'='*50}")

        plt.close('all')
        
        try:
            results = run_simulation_for_scenario(
                scenario_name, 
                params, 
                steps=steps,
                save_figures=save_figures
            )
            summary_results[scenario_name] = results
            print(f"[DONE] Completed scenario {i}/{total_scenarios}: {scenario_name}")
            print(f"  - Saved plots to figures/{scenario_name.lower().replace(' ', '_')}_*.png")
        except Exception as e:
            print(f"[ERROR] Failed to run scenario {scenario_name}: {str(e)}")
            print("Continuing with next scenario...")
            continue
        finally:
            plt.close('all')
    
    if generate_summary and summary_results:
        try:
            print("\nGenerating comparison figures across all scenarios...")
            generate_comparison_figures(summary_results)
        except Exception as e:
            print(f"[ERROR] Failed to generate comparison figures: {str(e)}")
    
    print("\nAll scenarios completed successfully!")
    print(f"Results saved to the 'figures' directory: {figures_dir.resolve()}")
    
    if save_figures:
        print("\nTo view the figures, check the 'figures' directory.")
    else:
        print("\nDisplaying summary plots (close window to exit)...")
        plt.show()
    
    return summary_results

def generate_comparison_figures(results):
    """Generate comparison figures across all scenarios."""
    if not results:
        print("No results to generate comparison figures")
        return
    
    scenario_names = list(results.keys())
    
    # Circularity Rates Comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(scenario_names))
    width = 0.35
    
    recycling_rates = [results[s]['recycling_rate'] * 100 for s in scenario_names]
    second_life_rates = [results[s]['second_life_rate'] * 100 for s in scenario_names]
    
    plt.bar(x - width/2, recycling_rates, width, label='Recycling Rate')
    plt.bar(x + width/2, second_life_rates, width, label='Second-Life Rate')
    
    for i, v in enumerate(recycling_rates):
        plt.text(i - width/2, v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
    
    for i, v in enumerate(second_life_rates):
        plt.text(i + width/2, v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
    
    plt.title('Circularity Rates Comparison')
    plt.ylabel('Rate (%)')
    plt.xticks(x, scenario_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig("figures/comparison_circularity_rates.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Material Recovery Comparison
    plt.figure(figsize=(12, 8))
    total_materials = [results[s]['total_materials_recovered'] for s in scenario_names]
    
    plt.bar(scenario_names, total_materials)
    plt.title('Total Material Recovery Comparison')
    plt.ylabel('Total Materials Recovered (kg)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig("figures/comparison_material_recovery.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Grid Storage Comparison
    plt.figure(figsize=(12, 8))
    grid_storage = [results[s]['grid_storage'] for s in scenario_names]
    
    plt.bar(scenario_names, grid_storage)
    plt.title('Grid Storage Capacity Comparison')
    plt.ylabel('Grid Storage Capacity (kWh)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig("figures/comparison_grid_storage.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Facility Utilization Comparison
    plt.figure(figsize=(12, 8))
    x = np.arange(len(scenario_names))
    width = 0.35
    
    recycler_util = [results[s]['facility_metrics']['recycler_capacity_utilization'] * 100 for s in scenario_names]
    refurbisher_util = [results[s]['facility_metrics']['refurbisher_capacity_utilization'] * 100 for s in scenario_names]
    
    plt.bar(x - width/2, recycler_util, width, label='Recycling Facilities')
    plt.bar(x + width/2, refurbisher_util, width, label='Refurbishment Facilities')
    
    for i, v in enumerate(recycler_util):
        plt.text(i - width/2, v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
    
    for i, v in enumerate(refurbisher_util):
        plt.text(i + width/2, v/2, f"{v:.1f}%", ha='center', color='white', fontweight='bold')
    
    plt.title('Facility Utilization Comparison')
    plt.ylabel('Utilization Rate (%)')
    plt.xticks(x, scenario_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    plt.savefig("figures/comparison_facility_utilization.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Generated comparison figures across all scenarios")

if __name__ == "__main__":
    print("EV Battery Circularity Simulation")
    print("=================================")
    print("This simulation models the lifecycle of EV batteries, from new to recycled/refurbished.")
    print("The model includes:")
    print("  - Battery owners with various environmental consciousness levels")
    print("  - Car manufacturers that produce batteries and take back end-of-life batteries")
    print("  - Recycling facilities that process batteries and recover materials")
    print("  - Refurbishing facilities that convert batteries for second-life applications")
    
    # Run all scenarios with graphs saved to the figures directory
    print("\nStarting simulation runs...")
    try:
        run_all_scenarios(steps=500, save_figures=True)
        print("\nSimulation complete. Check the 'figures' directory for all saved plots.")
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user. Partial results may be available in the 'figures' directory.")
    except Exception as e:
        print(f"\nSimulation failed with error: {str(e)}")
        print("Please check the error message above and try again.")