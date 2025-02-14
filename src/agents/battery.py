from dataclasses import dataclass
from enum import Enum
from typing import Optional

# This is an enum that is used to store the status of the battery in the lifecycle.
class BatteryStatus(Enum):
    """Status of a battery in the lifecycle."""
    NEW = "new"
    IN_USE = "in_use"
    END_OF_LIFE = "end_of_life"
    REFURBISHED = "refurbished"
    RECYCLED = "recycled"

class Battery:
    """Represents an EV battery in the simulation.
    
    Attributes:
        id (str): Unique identifier for the battery
        age (int): Age of the battery in months
        capacity (float): Current capacity in kWh
        health (float): Battery health percentage (0-1)
        status (BatteryStatus): Current status in lifecycle
        original_owner (EVOwner): Original owner of the battery
    """
    
    def __init__(self, battery_id: str, initial_capacity: float, 
                 original_owner: 'EVOwner') -> None:
        """Initialize a new battery.
        
        Args:
            battery_id (str): Unique identifier for the battery
            initial_capacity (float): Initial capacity in kWh
            original_owner (EVOwner): Original owner of the battery
        """
        self.id = battery_id
        self.age = 0
        self.capacity = initial_capacity
        self.health = 1.0
        self.status = BatteryStatus.NEW
        self.original_owner = original_owner
        
    def degrade_battery(self) -> None:
        """Simulate battery degradation over time."""
        pass
        
    def assess_condition(self) -> float:
        """Assess the current condition of the battery.
        
        Returns:
            float: Assessment score (0-1)
        """
        pass 