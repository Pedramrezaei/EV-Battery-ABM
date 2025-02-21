from .agent import Agent
from .battery import Battery, BatteryStatus
import random
from .eVOwner import EVOwner
from typing import Optional

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
                 production_capacity: int,
                 recycling_commitment: float,
                 warranty_age_limit: int = 96,  # 8 years default warranty
                 warranty_health_threshold: float = 0.7,
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
            initial_capacity=75.0,  # kWh, typical EV battery
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
        # Check age limit
        if battery.age > self.warranty_age_limit:
            return False
            
        # Check health threshold (for abnormal degradation)
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
        # Only take back end-of-life batteries
        if battery.status != BatteryStatus.END_OF_LIFE:
            return False
            
        # Check warranty status
        under_warranty = self.is_under_warranty(battery)
        
        # Always accept under warranty, otherwise based on recycling commitment
        if under_warranty or random.random() < self.recycling_commitment:
            self.batteries_taken_back += 1
            battery.status = BatteryStatus.RECYCLED
            return True
            
        return False
        
    def update_warranty_policy(self) -> None:
        """Update warranty policy based on battery performance and business conditions."""
        # Adjust warranty thresholds based on taken back batteries
        if self.batteries_taken_back > self.production_capacity * 0.2:  # Too many returns
            self.warranty_age_limit = max(60, self.warranty_age_limit - 12)  # Reduce by 1 year, min 5 years
            self.warranty_health_threshold = min(0.8, self.warranty_health_threshold + 0.05)
        else:  # Few returns, could be more generous
            self.warranty_age_limit = min(120, self.warranty_age_limit + 6)  # Increase by 6 months, max 10 years
            self.warranty_health_threshold = max(0.6, self.warranty_health_threshold - 0.02)
        
    def step(self) -> None:
        """Execute one step of the Car Manufacturer agent."""
        # Update warranty policy based on performance
        self.update_warranty_policy()
        
        # Reset production counter each step
        self.batteries_produced = 0 