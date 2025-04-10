from .agent import Agent
from .battery import Battery, BatteryStatus
import random
from .eVOwner import EVOwner

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.utils.constants import (
    DEFAULT_PRODUCTION_CAPACITY,
    DEFAULT_RECYCLING_COMMITMENT,
    DEFAULT_WARRANTY_AGE_LIMIT,
    DEFAULT_WARRANTY_HEALTH_THRESHOLD,
    DEFAULT_BATTERY_CAPACITY
)

class CarManufacturer(Agent):
    """Car Manufacturer agent that produces and handles batteries.
    
    Attributes:
        production_capacity (int): Number of batteries that can be produced per step
        recycling_commitment (float): Level of commitment to recycling (0-1)
        warranty_age_limit (int): Maximum battery age (months) covered by warranty
        warranty_health_threshold (float): Minimum health level covered by warranty
        batteries_produced (int): Counter for total batteries produced
        batteries_taken_back (int): Counter for batteries taken back
    """
    
    def __init__(self, 
                 unique_id: str,
                 model,
                 production_capacity: int = DEFAULT_PRODUCTION_CAPACITY,
                 recycling_commitment: float = DEFAULT_RECYCLING_COMMITMENT,
                 warranty_age_limit: int = DEFAULT_WARRANTY_AGE_LIMIT,
                 warranty_health_threshold: float = DEFAULT_WARRANTY_HEALTH_THRESHOLD,
                 x: float = 0.0,
                 y: float = 0.0) -> None:
        """Initialize a new Car Manufacturer.
        
        Args:
            unique_id (str): Unique identifier for the agent
            model: The model instance this agent is part of
            production_capacity (int): Maximum batteries that can be produced per step
            recycling_commitment (float): Level of commitment to recycling (0-1)
            warranty_age_limit (int): Maximum battery age (months) covered by warranty
            warranty_health_threshold (float): Minimum health level covered by warranty
            x (float): Initial x coordinate
            y (float): Initial y coordinate
        """
        super().__init__(unique_id, model, x, y)
        self.production_capacity = production_capacity
        self.recycling_commitment = recycling_commitment
        self.warranty_age_limit = warranty_age_limit
        self.warranty_health_threshold = warranty_health_threshold
        self.batteries_produced = 0
        self.batteries_taken_back = 0
        
    def produce_battery(self, owner: 'EVOwner') -> Battery:
        """Produce a new battery for an owner.
        
        Args:
            owner (EVOwner): The owner who will receive the battery
            
        Returns:
            Battery: The newly produced battery
        """
        if self.batteries_produced >= self.production_capacity:
            return None
            
        battery_id = f"BAT_{self.unique_id}_{self.batteries_produced}"
        battery = Battery(
            battery_id=battery_id,
            initial_capacity=DEFAULT_BATTERY_CAPACITY,  # kWh
            original_owner=owner
        )
        self.batteries_produced += 1
        return battery
        
    def is_under_warranty(self, battery: Battery) -> bool:
        """Check if a battery is still under warranty.
        
        Args:
            battery (Battery): The battery to check
            
        Returns:
            bool: True if battery is under warranty
        """

        if battery.age > self.warranty_age_limit:
            return False
            

        if battery.health < self.warranty_health_threshold:
            return False
            
        return True
        
    def handle_take_back(self, battery: Battery) -> bool:
        """Handle a battery that's been returned.
        
        Args:
            battery (Battery): The battery being returned
            
        Returns:
            bool: True if battery was accepted
        """
        # only take back EOL batteries
        if battery.status != BatteryStatus.END_OF_LIFE:
            print(f"Manufacturer {self.unique_id} rejected battery {battery.id}: Not end-of-life (status: {battery.status.value})")
            return False
            
        # check warranty
        under_warranty = self.is_under_warranty(battery)
        
        # warranty altijd, anders recycling commitment
        if under_warranty:
            print(f"Manufacturer {self.unique_id} accepted battery {battery.id}: Under warranty")
            self.batteries_taken_back += 1
            battery.change_status(BatteryStatus.COLLECTED)
        elif random.random() < self.recycling_commitment:
            print(f"Manufacturer {self.unique_id} accepted battery {battery.id}: Due to recycling commitment")
            self.batteries_taken_back += 1
            battery.change_status(BatteryStatus.COLLECTED)
        else:
            print(f"Manufacturer {self.unique_id} rejected battery {battery.id}: Not under warranty and failed recycling commitment check")
            return False
            

        forwarding_success = False
        max_attempts = 3  # try forwarding 3 times
        
        for attempt in range(max_attempts):
            if battery.is_suitable_for_second_life():
                print(f"Manufacturer {self.unique_id} forwarding battery {battery.id} to refurbisher (health: {battery.health:.2f})")
                forwarding_success = self.model.forward_to_refurbisher(battery)
                if forwarding_success:
                    break
                else:
                    print(f"Attempt {attempt+1}/{max_attempts}: Failed to forward to refurbisher, trying other options...")
                    # refurbishing failed -> recycling
                    if attempt == max_attempts - 2:  # last attempt, try recycling
                        break
            else:
                print(f"Manufacturer {self.unique_id} forwarding battery {battery.id} to recycler (health: {battery.health:.2f})")
                forwarding_success = self.model.forward_to_recycler(battery)
                if forwarding_success:
                    break
                else:
                    print(f"Attempt {attempt+1}/{max_attempts}: Failed to forward to recycler, trying again...")
        
        # if all forwarding failed, report the issue but still return True
        # because the manufacturer did accept the battery
        if not forwarding_success:
            print(f"WARNING: Manufacturer {self.unique_id} accepted battery {battery.id} but failed to forward it after {max_attempts} attempts")
            # last resort, try recycling
            if battery.is_suitable_for_second_life():
                print(f"Last resort: Trying to recycle battery {battery.id} that was suitable for second life")
                forwarding_success = self.model.forward_to_recycler(battery)
                
            # if still unsuccessful, the model will handle it as a stranded battery
            if not forwarding_success:
                self.model.track_stranded_battery(battery)
                
        return True
        
    def update_warranty_policy(self) -> None:
        """Update warranty policy based on battery performance and business conditions."""
        # adjust warranty thresholds
        if self.batteries_taken_back > self.production_capacity * 0.2:
            self.warranty_age_limit = max(60, self.warranty_age_limit - 12)
            self.warranty_health_threshold = min(0.8, self.warranty_health_threshold + 0.05)
        else:  # few returns, could be more generous
            self.warranty_age_limit = min(120, self.warranty_age_limit + 6)
            self.warranty_health_threshold = max(0.6, self.warranty_health_threshold - 0.02)
        
    def step(self) -> None:
        """Execute one step of the Car Manufacturer agent."""

        self.update_warranty_policy()
        

        self.batteries_produced = 0 