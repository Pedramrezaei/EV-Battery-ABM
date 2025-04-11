"""
Sensitivity analysis module for the battery circularity model.
This module provides functions to analyze how model outputs change with parameter variations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import random

# Add root directory to path
sys.path.append(str(Path(__file__).parents[2]))

from src.models.battery_circularity_model import BatteryCircularityModel
from src.agents.battery import Battery
from src.utils.constants import (
    DEFAULT_TECHNICAL_CAPABILITY,
    DEFAULT_REFURBISHER_CAPACITY,
    DEFAULT_RECYCLING_COMMITMENT,
    BATTERY_SECOND_LIFE_THRESHOLD,
    INCOME_RANGE,
    ENVIRONMENTAL_CONSCIOUSNESS_RANGE,
    DEFAULT_EFFICIENCY_RATE,
    DEFAULT_DEGRADATION_RATE
)

def run_sensitivity_analysis(steps=500, replications=10, base_seed=42):
    """
    Run sensitivity analysis on battery circularity model, varying one parameter at a time.
    
    Parameters:
    - steps: Number of time steps to run each simulation
    - replications: Number of replications for each parameter set (for statistical robustness)
    - base_seed: Base random seed to use (will increment for each replication)
    
    Returns:
    - DataFrame containing sensitivity analysis results for each parameter and metric
    """
    
    # Parameters to analyze with their baseline values
    parameters = {
        'Technical capability': DEFAULT_TECHNICAL_CAPABILITY,
        'Refurbisher capacity': DEFAULT_REFURBISHER_CAPACITY,
        'Recycling commitment': DEFAULT_RECYCLING_COMMITMENT,
        'Second-life threshold': BATTERY_SECOND_LIFE_THRESHOLD,
        'Environmental consciousness': np.mean(ENVIRONMENTAL_CONSCIOUSNESS_RANGE),
        'Degradation rate': DEFAULT_DEGRADATION_RATE,
        'Recycling efficiency': DEFAULT_EFFICIENCY_RATE
    }
    
    # Save original class attribute values for proper reset
    original_second_life_threshold = Battery.SECOND_LIFE_THRESHOLD
    
    # Variation percentages to test
    variations = [-0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20]
    
    # Output metrics to track
    metrics = [
        'Recycling_Rate', 
        'Second_Life_Rate', 
        'Materials_Recovered',
        'Total_Grid_Storage'
    ]
    
    # Initialize results dictionary
    results = {
        'Parameter': [],
        'Variation': [],
        'Recycling_Rate': [],
        'Second_Life_Rate': [],
        'Materials_Recovered': [],
        'Total_Grid_Storage': []
    }
    
    # Add dictionaries for absolute values
    absolute_baseline = {metric: 0 for metric in metrics}
    absolute_values = {metric: [] for metric in metrics}
    
    # Baseline run to compare against
    print("Running baseline model...")
    baseline_results = run_baseline_model(steps, replications, metrics, base_seed)
    
    # Store absolute baseline values
    for metric in metrics:
        absolute_baseline[metric] = baseline_results[metric]
        print(f"  Baseline {metric}: {baseline_results[metric]:.6f}")
    
    # For each parameter
    for param_name, baseline_value in parameters.items():
        print(f"Analyzing sensitivity for {param_name}...")
        
        # For each variation percentage
        for var_pct in variations:
            # Skip baseline case (already computed)
            if var_pct == 0:
                results['Parameter'].append(param_name)
                results['Variation'].append(0)
                for metric in metrics:
                    results[metric].append(0)  # 0% change from baseline
                    absolute_values[metric].append(absolute_baseline[metric])
                continue
                
            # Calculate new parameter value
            param_value = baseline_value * (1 + var_pct)
            
            # Ensure the parameter value makes sense (e.g., probabilities stay between 0-1)
            if (param_name in ['Technical capability', 'Recycling commitment', 'Second-life threshold', 
                              'Environmental consciousness', 'Recycling efficiency'] and 
                (param_value < 0 or param_value > 1)):
                param_value = max(0, min(1, param_value))
            
            # For capacity parameters, ensure they're integers ≥ 1
            if param_name in ['Refurbisher capacity']:
                param_value = max(1, int(param_value))
            
            # Run model with modified parameter
            print(f"  Running with {param_name} = {param_value:.6f} ({var_pct * 100:+.0f}% change)")
            metric_values = run_model_with_parameter(param_name, param_value, steps, replications, metrics, base_seed)
            
            # Print absolute values for comparison
            for metric in metrics:
                print(f"    {metric}: {metric_values[metric]:.6f} (baseline: {absolute_baseline[metric]:.6f})")
            
            # Add parameter and variation once per combination
            results['Parameter'].append(param_name)
            results['Variation'].append(var_pct * 100)  # Convert to percentage
            
            # Calculate percentage changes from baseline
            for metric in metrics:
                if absolute_baseline[metric] == 0:
                    pct_change = 0  # Avoid division by zero
                    print(f"    WARNING: Baseline value for {metric} is zero. Percentage change is undefined.")
                else:
                    pct_change = ((metric_values[metric] - absolute_baseline[metric]) / absolute_baseline[metric]) * 100
                
                results[metric].append(round(pct_change, 1))  # Rounded percentage change
                absolute_values[metric].append(metric_values[metric])
                
            # Reset class attributes to original values
            Battery.SECOND_LIFE_THRESHOLD = original_second_life_threshold
    
    # Convert to DataFrame
    sensitivity_df = pd.DataFrame(results)
    
    # Create DataFrame with absolute values for reference
    absolute_df = pd.DataFrame({
        'Parameter': results['Parameter'],
        'Variation': results['Variation'],
    })
    for metric in metrics:
        absolute_df[metric] = absolute_values[metric]
    
    return sensitivity_df, absolute_df

def run_baseline_model(steps, replications, metrics, base_seed=42):
    """Run the model with baseline parameters and return average metric values."""
    # Dictionary to store metric results
    metric_results = {metric: 0 for metric in metrics}
    
    # Run replications
    for rep in range(replications):
        # Set fixed seed for reproducibility
        seed = base_seed + rep
        print(f"  Baseline replication {rep+1}/{replications} (seed: {seed})")
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Create new model with the seed
        model = BatteryCircularityModel()
        
        # Run for specified steps
        for step in range(steps):
            model.step()
            
            # Print progress every 100 steps
            if step > 0 and step % 100 == 0:
                print(f"    Step {step}/{steps}")
        
        # Collect final metric values
        for metric in metrics:
            if metric == 'Materials_Recovered':
                # For materials, sum up all materials recovered
                materials = model.get_total_materials_recovered()
                material_sum = sum(materials.values())
                metric_results[metric] += material_sum
                print(f"    Materials recovered: {material_sum:.6f}")
            else:
                # For other metrics, get the value from the appropriate method
                value = 0
                if metric == 'Recycling_Rate':
                    value = model.calculate_recycling_rate()
                elif metric == 'Second_Life_Rate':
                    value = model.calculate_second_life_rate()
                elif metric == 'Total_Grid_Storage':
                    value = model.calculate_total_grid_storage()
                
                metric_results[metric] += value
                print(f"    {metric}: {value:.6f}")
    
    # Average the results across replications
    for metric in metrics:
        metric_results[metric] /= replications
    
    return metric_results

def run_model_with_parameter(param_name, param_value, steps, replications, metrics, base_seed=42):
    """Run the model with a specific parameter value and return metric values."""
    # Dictionary to store metric results
    metric_results = {metric: 0 for metric in metrics}
    
    # Save original class attribute values
    original_second_life_threshold = Battery.SECOND_LIFE_THRESHOLD
    
    # Run replications
    for rep in range(replications):
        # Set fixed seed for reproducibility
        seed = base_seed + rep
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize model
        model = BatteryCircularityModel()
        
        # Apply parameter modification based on parameter name
        if param_name == 'Technical capability':
            for refurbisher in model.refurbishers:
                refurbisher.technical_capability = param_value
        elif param_name == 'Refurbisher capacity':
            for refurbisher in model.refurbishers:
                refurbisher.capacity = param_value
        elif param_name == 'Recycling commitment':
            for manufacturer in model.manufacturers:
                manufacturer.recycling_commitment = param_value
        elif param_name == 'Second-life threshold':
            # This modifies the threshold for all batteries
            Battery.SECOND_LIFE_THRESHOLD = param_value
            # Also update any existing batteries' thresholds
            for battery in model.all_batteries:
                # Ensure any instance-specific overrides also get updated
                if hasattr(battery, 'SECOND_LIFE_THRESHOLD'):
                    battery.SECOND_LIFE_THRESHOLD = param_value
        elif param_name == 'Environmental consciousness':
            for owner in model.owners:
                owner.environmental_consciousness = param_value
        elif param_name == 'Degradation rate':
            # Apply to new batteries and existing batteries
            for battery in model.all_batteries:
                battery.degradation_rate = param_value
        elif param_name == 'Recycling efficiency':
            for recycler in model.recyclers:
                recycler.efficiency_rate = param_value
        
        # Run the model for the specified steps
        for step in range(steps):
            model.step()
            
            # Print progress every 100 steps
            if step > 0 and step % 100 == 0 and rep == 0:  # Only print for first replication
                print(f"    Step {step}/{steps}")
        
        # Collect final metric values
        for metric in metrics:
            if metric == 'Materials_Recovered':
                materials = model.get_total_materials_recovered()
                metric_results[metric] += sum(materials.values())
            else:
                if metric == 'Recycling_Rate':
                    metric_results[metric] += model.calculate_recycling_rate()
                elif metric == 'Second_Life_Rate':
                    metric_results[metric] += model.calculate_second_life_rate()
                elif metric == 'Total_Grid_Storage':
                    metric_results[metric] += model.calculate_total_grid_storage()
    
    # Average the results across replications
    for metric in metrics:
        metric_results[metric] /= replications
    
    # Reset class attributes to original values
    Battery.SECOND_LIFE_THRESHOLD = original_second_life_threshold
    
    return metric_results

def generate_sensitivity_heatmap(sensitivity_df, output_metric, output_dir="figures"):
    """Generate a heatmap visualization of sensitivity analysis results for a given metric."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Reshape data for heatmap
    pivot_data = sensitivity_df.pivot(
        index="Parameter", 
        columns="Variation", 
        values=output_metric
    )
    
    # Sort parameters by average absolute sensitivity
    param_sensitivity = pivot_data.abs().mean(axis=1).sort_values(ascending=False)
    sorted_params = param_sensitivity.index.tolist()
    pivot_data = pivot_data.loc[sorted_params]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap with colormap that goes from red (negative) to white (neutral) to green (positive)
    cmap = sns.diverging_palette(10, 120, as_cmap=True)
    
    # Determine maximum absolute value for symmetric color scaling
    vmax = max(pivot_data.abs().max().max(), 10)  # At least ±10%
    
    # Create heatmap
    sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap=cmap,
        vmin=-vmax, 
        vmax=vmax,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "% Change in " + output_metric.replace('_', ' ')}
    )
    
    # Add labels
    plt.title(f"Sensitivity Analysis: Impact on {output_metric.replace('_', ' ')}")
    plt.xlabel("Parameter Variation (%)")
    plt.ylabel("Model Parameter")
    
    # Add note about percentage change
    plt.figtext(0.5, 0.01, "Values represent percentage change from baseline", 
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    output_file = f"{output_dir}/sensitivity_{output_metric}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to {output_file}")
    plt.close()

def save_all_sensitivity_data(df, df_absolute, output_dir="figures"):
    """Save the sensitivity analysis results to CSV and generate visualizations."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save raw data
    output_file = f"{output_dir}/sensitivity_analysis_results.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")
    
    # Save absolute values
    output_file_abs = f"{output_dir}/sensitivity_analysis_absolute.csv"
    df_absolute.to_csv(output_file_abs, index=False)
    print(f"Saved absolute values to {output_file_abs}")
    
    # Generate heatmaps for each metric
    for metric in ['Recycling_Rate', 'Second_Life_Rate', 'Materials_Recovered', 'Total_Grid_Storage']:
        generate_sensitivity_heatmap(df, metric, output_dir)
    
    # Create combined heatmap
    generate_combined_heatmap(df, output_dir)

def generate_combined_heatmap(df, output_dir="figures"):
    """Generate a combined heatmap showing overall sensitivity across metrics."""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get unique parameters and variations
    parameters = df['Parameter'].unique()
    variations = sorted(df['Variation'].unique())
    
    # Calculate normalized impact across all metrics
    metrics = ['Recycling_Rate', 'Second_Life_Rate', 'Materials_Recovered', 'Total_Grid_Storage']
    
    # Create a new DataFrame for the combined impact
    combined_data = []
    
    for param in parameters:
        for var in variations:
            param_var_df = df[(df['Parameter'] == param) & (df['Variation'] == var)]
            if not param_var_df.empty:
                # Calculate normalized impact (average of absolute percentage changes)
                impact = 0
                for metric in metrics:
                    impact += abs(param_var_df[metric].values[0]) / len(metrics)
                
                combined_data.append({
                    'Parameter': param,
                    'Variation': var,
                    'Combined_Impact': round(impact, 1)
                })
    
    combined_df = pd.DataFrame(combined_data)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    pivot_data = combined_df.pivot(
        index="Parameter", 
        columns="Variation", 
        values="Combined_Impact"
    )
    
    # Sort by average sensitivity
    param_sensitivity = pivot_data.mean(axis=1).sort_values(ascending=False)
    sorted_params = param_sensitivity.index.tolist()
    pivot_data = pivot_data.loc[sorted_params]
    
    # Create heatmap with sequential colormap (more intense color = higher impact)
    sns.heatmap(
        pivot_data, 
        annot=True, 
        cmap="YlOrRd",
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={"label": "Average Absolute % Change Across All Metrics"}
    )
    
    plt.title("Overall Parameter Sensitivity Analysis")
    plt.xlabel("Parameter Variation (%)")
    plt.ylabel("Model Parameter")
    
    plt.figtext(0.5, 0.01, "Values represent average absolute percentage change across all metrics", 
                ha="center", fontsize=10, style='italic')
    
    plt.tight_layout()
    output_file = f"{output_dir}/sensitivity_combined.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved combined heatmap to {output_file}")
    plt.close()

if __name__ == "__main__":
    # Create output directory
    output_dir = "figures/sensitivity"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("Starting sensitivity analysis...")
    print("NOTE: This will take longer with improved settings for better accuracy")
    sensitivity_results, absolute_values = run_sensitivity_analysis(steps=500, replications=10, base_seed=42)
    save_all_sensitivity_data(sensitivity_results, absolute_values, output_dir)
    print("Sensitivity analysis complete. Results saved to CSV and visualizations generated.") 