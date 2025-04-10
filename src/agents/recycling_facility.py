from .agent import Agent
from .battery import Battery, BatteryStatus
from typing import List, Dict
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.utils.constants import (
    DEFAULT_PROCESSING_CAPACITY,
    DEFAULT_EFFICIENCY_RATE,
    BATTERY_MATERIALS
)

class RecyclingFacility(Agent):
    """Recycling Facility agent that processes end-of-life batteries.
    
    Attributes:
        processing_capacity (int): Number of batteries that can be processed per step
        efficiency_rate (float): Efficiency of material recovery (0-1)
        current_inventory (List[Battery]): Batteries currently held for processing
        materials_recovered (Dict): Tracking of recovered materials
        total_processed (int): Total number of batteries processed
        total_materials (Dict): Total materials recovered over time
    """
    
    def __init__(self, 
                 unique_id: str,
                 model,
                 processing_capacity: int = DEFAULT_PROCESSING_CAPACITY,
                 efficiency_rate: float = DEFAULT_EFFICIENCY_RATE,
                 x: float = 0.0,
                 y: float = 0.0) -> None:
        """Initialize a new Recycling Facility."""
        super().__init__(unique_id, model, x, y)
        self.processing_capacity = processing_capacity
        self.efficiency_rate = efficiency_rate
        self.current_inventory = []
        self.materials_recovered = {
            'lithium': 0.0,
            'cobalt': 0.0,
            'nickel': 0.0,
            'copper': 0.0
        }
        self.total_processed = 0
        self.total_materials = {
            'lithium': 0.0,
            'cobalt': 0.0,
            'nickel': 0.0,
            'copper': 0.0
        }
        
        # Add new attributes for more realistic processing
        self.currently_processing = []  # Batteries being processed
        self.processing_times = {}  # Time remaining for each battery being processed
        self.operational = True  # Whether the facility is operational
        self.downtime_counter = 0  # Counter for facility downtime
        
    def receive_battery(self, battery: Battery) -> bool:
        """Add a battery to the facility's inventory.
        
        Args:
            battery (Battery): Battery to be recycled
            
        Returns:
            bool: True if battery was accepted
        """
        # Check if facility is operational
        if not self.operational:
            print(f"Recycler {self.unique_id} rejected battery {battery.id}: Facility is temporarily offline")
            return False
            
        # Check inventory capacity - now consider both inventory and processing
        total_batteries = len(self.current_inventory) + len(self.currently_processing)
        if total_batteries >= self.processing_capacity * 2:  # Buffer size
            print(f"Recycler {self.unique_id} rejected battery {battery.id}: Inventory full ({total_batteries}/{self.processing_capacity * 2})")
            return False
            
        # Accept either END_OF_LIFE or COLLECTED batteries
        if battery.status not in [BatteryStatus.END_OF_LIFE, BatteryStatus.COLLECTED]:
            print(f"Recycler {self.unique_id} rejected battery {battery.id}: Incorrect status {battery.status.value}")
            return False
            
        self.current_inventory.append(battery)
        print(f"Recycler {self.unique_id} accepted battery {battery.id}. Inventory: {len(self.current_inventory)}, Processing: {len(self.currently_processing)}")
        
        # Ensure the model tracks that this battery is now in a recycling facility
        if hasattr(self.model, 'track_battery'):
            self.model.track_battery(battery)
            
        return True
        
    def start_processing(self, battery: Battery) -> None:
        """Start processing a battery with a realistic timeframe.
        
        Args:
            battery (Battery): Battery to process
        """
        # Remove from inventory and add to processing queue
        if battery in self.current_inventory:
            self.current_inventory.remove(battery)
            
        self.currently_processing.append(battery)
        
        # Determine processing time based on battery health and complexity
        # Lower health batteries take longer to process
        base_time = 1  # Minimum 1 month
        health_factor = max(1, int((1 - battery.health) * 3))  # More degraded = longer processing
        processing_time = base_time + health_factor
        
        self.processing_times[battery.id] = processing_time
        print(f"Recycler {self.unique_id} started processing battery {battery.id}, estimated time: {processing_time} months")
        
    def process_battery(self, battery: Battery) -> Dict[str, float]:
        """Complete processing of a battery for material recovery.
        
        Args:
            battery (Battery): Battery to process
            
        Returns:
            Dict[str, float]: Amount of each material recovered
        """
        print(f"Recycler {self.unique_id} completing processing of battery {battery.id} (health: {battery.health:.2f})")
        
        # Remove from processing queue
        if battery in self.currently_processing:
            self.currently_processing.remove(battery)
            
        # Clear processing time
        if battery.id in self.processing_times:
            del self.processing_times[battery.id]
        
        # Base material content (kg) for a typical EV battery
        base_materials = BATTERY_MATERIALS
        
        # Recovery affected by battery health and facility efficiency
        recovery_rate = self.efficiency_rate * (0.8 + 0.2 * battery.health)
        
        recovered = {}
        for material, amount in base_materials.items():
            recovered_amount = amount * recovery_rate
            self.materials_recovered[material] += recovered_amount
            self.total_materials[material] += recovered_amount
            recovered[material] = recovered_amount
            
        # Only the recycling facility should set RECYCLED status
        previous_status = battery.status
        battery.change_status(BatteryStatus.RECYCLED)
        self.total_processed += 1
        print(f"Recycler {self.unique_id} finished processing battery {battery.id} (was {previous_status.value}). Total processed: {self.total_processed}")
        return recovered
        
    def calculate_recovery_rate(self) -> Dict[str, float]:
        """Calculate the current recovery rate for materials.
        
        Returns:
            Dict[str, float]: Recovery rate for each material
        """
        if self.total_processed == 0:
            return {mat: 0.0 for mat in self.materials_recovered.keys()}
            
        return {
            material: amount / self.total_processed 
            for material, amount in self.total_materials.items()
        }
        
    def get_metrics(self) -> Dict[str, float]:
        """Get facility performance metrics.
        
        Returns:
            Dict containing:
            - total_processed: Number of batteries processed
            - current_inventory: Number of batteries in inventory
            - recovery_rates: Current recovery rates by material
            - total_materials: Total materials recovered
        """
        return {
            'total_processed': self.total_processed,
            'current_inventory': len(self.current_inventory),
            'recovery_rates': self.calculate_recovery_rate(),
            'total_materials': self.total_materials.copy()
        }
        
    def step(self) -> None:
        """Execute one step of the Recycling Facility agent."""
        # Random chance of facility downtime (5% chance per step)
        if self.operational and random.random() < 0.05:
            self.operational = False
            self.downtime_counter = random.randint(1, 3)  # 1-3 months of downtime
            print(f"Recycler {self.unique_id} is temporarily offline for {self.downtime_counter} months (maintenance)")
            return
            
        # If facility is down, decrease downtime counter
        if not self.operational:
            self.downtime_counter -= 1
            if self.downtime_counter <= 0:
                self.operational = True
                print(f"Recycler {self.unique_id} is back online after maintenance")
            else:
                print(f"Recycler {self.unique_id} remains offline for {self.downtime_counter} more months")
            return
            
        # Process batteries currently being processed
        completed = []
        batteries_to_remove = []
        for battery in list(self.currently_processing):
            # Check if battery has a processing time entry
            if battery.id not in self.processing_times:
                print(f"Warning: Battery {battery.id} has no processing time record. Setting default time.")
                self.processing_times[battery.id] = 2  # Default to 2 months processing time
            
            # Now safely decrease processing time
            self.processing_times[battery.id] -= 1
            
            # If processing is complete, mark for completion
            if self.processing_times[battery.id] <= 0:
                completed.append(battery)
                
        # Complete processing for batteries that are done
        for battery in completed:
            self.process_battery(battery)
            
        # Start processing new batteries up to capacity
        available_capacity = self.processing_capacity - len(self.currently_processing)
        
        if available_capacity > 0 and self.current_inventory:
            batteries_to_process = self.current_inventory[:available_capacity]
            
            if batteries_to_process:
                print(f"Recycler {self.unique_id} starting to process {len(batteries_to_process)} new batteries this step")
            
            for battery in batteries_to_process:
                self.start_processing(battery) 