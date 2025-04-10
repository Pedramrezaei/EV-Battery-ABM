import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import numpy as np
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider

from models.battery_circularity_model import BatteryCircularityModel
from agents.eVOwner import EVOwner
from agents.car_manufacturer import CarManufacturer
from agents.recycling_facility import RecyclingFacility
from agents.battery_refurbisher import BatteryRefurbisher

def agent_portrayal(agent):
    """Define how agents are portrayed in the visualization."""
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.5,
        "Layer": 0,
        "text": "",
        "text_color": "black"
    }

    if isinstance(agent, EVOwner):
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
        if agent.battery:
            portrayal["text"] = f"🔋{agent.battery.health:.1f}"
    elif isinstance(agent, CarManufacturer):
        portrayal["Color"] = "green"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 0.8
        portrayal["h"] = 0.8
        portrayal["Layer"] = 2
        portrayal["text"] = "🏭"
    elif isinstance(agent, RecyclingFacility):
        portrayal["Color"] = "red"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 0.8
        portrayal["h"] = 0.8
        portrayal["Layer"] = 2
        portrayal["text"] = "♻️"
    elif isinstance(agent, BatteryRefurbisher):
        portrayal["Color"] = "purple"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 0.8
        portrayal["h"] = 0.8
        portrayal["Layer"] = 2
        portrayal["text"] = "🔧"

    return portrayal

def create_model_visualization():
    """Create and return a server for model visualization."""
    # Grid visualization
    grid = CanvasGrid(agent_portrayal, 50, 50, 500, 500)

    # Charts
    battery_status_chart = ChartModule([
        {"Label": "Active_Batteries", "Color": "blue"},
        {"Label": "End_Of_Life", "Color": "red"},
        {"Label": "Recycled", "Color": "green"},
        {"Label": "Refurbished", "Color": "purple"}
    ])

    circularity_chart = ChartModule([
        {"Label": "Recycling_Rate", "Color": "green"},
        {"Label": "Second_Life_Rate", "Color": "purple"}
    ])

    materials_chart = ChartModule([
        {"Label": "Materials_Recovered", "Color": "orange"}
    ])

    # Model parameters
    model_params = {
        "num_owners": Slider(
            "Number of EV Owners", 100, 10, 500, 10
        ),
        "num_manufacturers": Slider(
            "Number of Manufacturers", 3, 1, 10, 1
        ),
        "num_recyclers": Slider(
            "Number of Recyclers", 2, 1, 5, 1
        ),
        "num_refurbishers": Slider(
            "Number of Refurbishers", 2, 1, 5, 1
        )
    }

    # Create and return server
    server = ModularServer(
        BatteryCircularityModel,
        [grid, battery_status_chart, circularity_chart, materials_chart],
        "Battery Circularity Model",
        model_params
    )

    return server

def plot_simulation_results(model: BatteryCircularityModel, steps: int):
    """Create plots from simulation results.
    
    Args:
        model (BatteryCircularityModel): The simulated model
        steps (int): Number of steps simulated
    """
    # Get the data from the datacollector
    data = model.datacollector.get_model_vars_dataframe()

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Battery Circularity Simulation Results')

    # Battery status plot
    status_cols = ['Active_Batteries', 'End_Of_Life', 'Recycled', 'Refurbished']
    data[status_cols].plot(ax=axes[0, 0])
    axes[0, 0].set_title('Battery Status Over Time')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Number of Batteries')

    # Circularity rates plot
    rate_cols = ['Recycling_Rate', 'Second_Life_Rate']
    data[rate_cols].plot(ax=axes[0, 1])
    axes[0, 1].set_title('Circularity Rates')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Rate')

    # Materials recovered plot
    materials = data['Materials_Recovered'].iloc[-1]
    axes[1, 0].bar(materials.keys(), materials.values())
    axes[1, 0].set_title('Total Materials Recovered')
    axes[1, 0].set_xlabel('Material')
    axes[1, 0].set_ylabel('Amount (kg)')
    plt.xticks(rotation=45)

    # Battery age distribution
    ages = [agent.battery.age for agent in model.schedule.agents 
            if isinstance(agent, EVOwner) and agent.battery]
    sns.histplot(ages, ax=axes[1, 1], bins=20)
    axes[1, 1].set_title('Battery Age Distribution')
    axes[1, 1].set_xlabel('Age (months)')
    axes[1, 1].set_ylabel('Count')

    plt.tight_layout()
    return fig 