from .agent import Agent
from .battery import Battery, BatteryStatus
from typing import List, Dict
import random

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.utils.constants import (
    DEFAULT_TECHNICAL_CAPABILITY,
    DEFAULT_REFURBISHER_CAPACITY,
    DEFAULT_GRID_STORAGE_FACTOR
)

class BatteryRefurbisher(Agent):
    """Battery Refurbisher agent that converts batteries for second-life applications.
    
    Attributes:
        technical_capability (float): Technical expertise level (0-1)
        capacity (int): Number of batteries that can be processed per step
        inventory (List[Battery]): Batteries currently held for refurbishment
        successful_conversions (int): Counter for successful refurbishments
    """
    
    def __init__(self, 
                 unique_id: str,
                 model,
                 technical_capability: float = DEFAULT_TECHNICAL_CAPABILITY,
                 capacity: int = DEFAULT_REFURBISHER_CAPACITY,
                 x: float = 0.0,
                 y: float = 0.0) -> None:
        """Initialize a new Battery Refurbisher."""
        super().__init__(unique_id, model, x, y)
        self.technical_capability = technical_capability
        self.capacity = capacity
        self.inventory = []
        self.successful_conversions = 0
        

        self.currently_processing = []
        self.processing_times = {}
        self.operational = True
        self.downtime_counter = 0
        self.refurbished_batteries = []
        
    def receive_battery(self, battery: Battery) -> bool:
        """Add a battery to the refurbisher's inventory.
        
        Args:
            battery (Battery): Battery to be refurbished
            
        Returns:
            bool: True if battery was accepted
        """
        # Check voor operational
        if not self.operational:
            print(f"Refurbisher {self.unique_id} rejected battery {battery.id}: Facility is temporarily offline")
            return False
            
        # Check  inventory(inventory en processing beide)
        total_batteries = len(self.inventory) + len(self.currently_processing)
        if total_batteries >= self.capacity * 2:  # Buffer size
            print(f"Refurbisher {self.unique_id} rejected battery {battery.id}: Inventory full ({total_batteries}/{self.capacity * 2})")
            return False
            

        if not battery.is_suitable_for_second_life():
            print(f"Refurbisher {self.unique_id} rejected battery {battery.id}: Not suitable for second life (health: {battery.health:.2f})")
            # not suitable? -> recycling
            if hasattr(self.model, 'forward_to_recycler'):
                print(f"Refurbisher {self.unique_id} is forwarding battery {battery.id} to recycling instead")
                return self.model.forward_to_recycler(battery)
            return False
            
        self.inventory.append(battery)
        print(f"Refurbisher {self.unique_id} accepted battery {battery.id}. Inventory: {len(self.inventory)}, Processing: {len(self.currently_processing)}")
        
        # track battery
        if hasattr(self.model, 'track_battery'):
            self.model.track_battery(battery)
            
        return True
        
    def start_processing(self, battery: Battery) -> None:
        """Start refurbishing a battery with a realistic timeframe.
        
        Args:
            battery (Battery): Battery to refurbish
        """

        if battery in self.inventory:
            self.inventory.remove(battery)
            
        self.currently_processing.append(battery)


        base_time = 2  # Minimum 2 months
        health_factor = max(1, int((Battery.GOOD_HEALTH_THRESHOLD - battery.health) * 5))
        
        # time based on technical capability
        tech_factor = max(1, int((1 - self.technical_capability) * 3))
        
        processing_time = base_time + health_factor + tech_factor
        
        self.processing_times[battery.id] = processing_time
        print(f"Refurbisher {self.unique_id} started refurbishing battery {battery.id}, estimated time: {processing_time} months")
        
    def assess_battery(self, battery: Battery) -> float:
        """Assess a battery's suitability for refurbishment.
        
        Args:
            battery (Battery): Battery to assess
            
        Returns:
            float: Assessment score (0-1)
        """
        base_score = battery.health / Battery.GOOD_HEALTH_THRESHOLD
        expertise_factor = 0.8 + 0.2 * self.technical_capability
        
        assessment = min(1.0, base_score * expertise_factor)
        print(f"Refurbisher {self.unique_id} assessed battery {battery.id}: Score {assessment:.2f} (health: {battery.health:.2f})")
        return assessment
        
    def refurbish_battery(self, battery: Battery) -> bool:
        """Complete the refurbishment of a battery.
        
        Args:
            battery (Battery): Battery to refurbish
            
        Returns:
            bool: True if refurbishment was successful
        """
        print(f"Refurbisher {self.unique_id} completing refurbishment of battery {battery.id}")
        
        # weg uit processing queue
        if battery in self.currently_processing:
            self.currently_processing.remove(battery)

        if battery.id in self.processing_times:
            del self.processing_times[battery.id]
            
        # check successful
        assessment_score = self.assess_battery(battery)
        success_chance = assessment_score * self.technical_capability
        
        success = random.random() < success_chance
        if success:
            previous_status = battery.status
            battery.change_status(BatteryStatus.REFURBISHED)
            self.successful_conversions += 1
            self.refurbished_batteries.append(battery)
            print(f"Refurbisher {self.unique_id} successfully refurbished battery {battery.id} (was {previous_status.value}). Total refurbished: {self.successful_conversions}")
            return True
            
        print(f"Refurbisher {self.unique_id} failed to refurbish battery {battery.id}")
        
        #refurbishment fails -> recycle
        if hasattr(self.model, 'forward_to_recycler'):
            print(f"Refurbisher {self.unique_id} is forwarding failed battery {battery.id} to recycling")
            return self.model.forward_to_recycler(battery)
            
        return False
        
    def convert_to_grid_storage(self, batteries: List[Battery]) -> float:
        """Convert multiple batteries into grid storage capacity.
        
        Args:
            batteries (List[Battery]): Batteries to combine
            
        Returns:
            float: Total storage capacity achieved (kWh)
        """
        total_capacity = 0.0
        for battery in batteries:
            if battery.status == BatteryStatus.REFURBISHED:
                # Grid storage use 70-90% of  capacity
                usable_factor = DEFAULT_GRID_STORAGE_FACTOR[0] + (DEFAULT_GRID_STORAGE_FACTOR[1] - DEFAULT_GRID_STORAGE_FACTOR[0]) * self.technical_capability
                total_capacity += battery.capacity * usable_factor
        return total_capacity
        
    def step(self) -> None:
        """Execute one step of the Battery Refurbisher agent."""
        # random chance of downtime
        if self.operational and random.random() < 0.07:
            self.operational = False
            self.downtime_counter = random.randint(1, 4)
            print(f"Refurbisher {self.unique_id} is temporarily offline for {self.downtime_counter} months (maintenance/upgrades)")
            return
            
        # if down, decrease counter
        if not self.operational:
            self.downtime_counter -= 1
            if self.downtime_counter <= 0:
                self.operational = True
                print(f"Refurbisher {self.unique_id} is back online after maintenance")
            else:
                print(f"Refurbisher {self.unique_id} remains offline for {self.downtime_counter} more months")
            return
        

        completed = []
        for battery in list(self.currently_processing):
            # check for processing time entry
            if battery.id not in self.processing_times:
                print(f"Warning: Battery {battery.id} has no processing time record. Setting default time.")
                self.processing_times[battery.id] = 3  # Default to 3 months processing time

            self.processing_times[battery.id] -= 1

            if self.processing_times[battery.id] <= 0:
                completed.append(battery)
                
        # complete processing
        refurbished_this_step = []
        for battery in completed:
            if self.refurbish_battery(battery):
                refurbished_this_step.append(battery)
                
        # start processing new batteries
        available_capacity = self.capacity - len(self.currently_processing)
        
        if available_capacity > 0 and self.inventory:
            batteries_to_process = self.inventory[:available_capacity]
            
            if batteries_to_process:
                print(f"Refurbisher {self.unique_id} starting to refurbish {len(batteries_to_process)} new batteries this step")
            
            for battery in batteries_to_process:
                self.start_processing(battery)
                
        # convert refurbished batteries periodically
        # simulates the  process of preparing and installing grid storage systems
        if self.refurbished_batteries and (self.model.schedule.steps % random.randint(3, 6) == 0):
            storage_capacity = self.convert_to_grid_storage(self.refurbished_batteries)
            print(f"Refurbisher {self.unique_id} created {storage_capacity:.2f} kWh of grid storage from {len(self.refurbished_batteries)} batteries")
            self.refurbished_batteries = []  # Clear the batch 