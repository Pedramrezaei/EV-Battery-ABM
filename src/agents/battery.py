from enum import Enum
from datetime import datetime
from typing import TYPE_CHECKING, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.utils.constants import (
    BATTERY_GOOD_HEALTH_THRESHOLD,
    BATTERY_SECOND_LIFE_THRESHOLD,
    DEFAULT_DEGRADATION_RATE
)

if TYPE_CHECKING:
    from src.agents.eVOwner import EVOwner

class BatteryStatus(Enum):
    """Status of a battery in the lifecycle.
    
    Flow:
    NEW -> IN_USE -> END_OF_LIFE -> 
        -> COLLECTED (by manufacturer) ->
            -> RECYCLED (by recycling facility)
            -> REFURBISHED (by battery refurbisher)
    """
    NEW = "new"
    IN_USE = "in_use"
    END_OF_LIFE = "end_of_life"
    COLLECTED = "collected"
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
        manufacture_date (datetime): When the battery was manufactured
        initial_capacity (float): The original capacity when new
        cycle_count (int): Number of charge cycles
        degradation_rate (float): Rate at which battery degrades per cycle
        
    Health Thresholds:
        1.0 - 0.8: Good condition (NEW/IN_USE)
        0.8 - 0.6: End of first life, suitable for second life
        < 0.6: End of life, needs recycling
    """

    GOOD_HEALTH_THRESHOLD = BATTERY_GOOD_HEALTH_THRESHOLD
    SECOND_LIFE_THRESHOLD = BATTERY_SECOND_LIFE_THRESHOLD
    
    def __init__(self, 
                 battery_id: str, 
                 initial_capacity: float,
                 original_owner: 'EVOwner',
                 degradation_rate: float = DEFAULT_DEGRADATION_RATE) -> None:
        """Initialize a new battery.
        
        Args:
            battery_id (str): Unique identifier for the battery
            initial_capacity (float): Initial capacity in kWh
            original_owner (EVOwner): Original owner of the battery
            degradation_rate (float): Rate of capacity loss per cycle (default 0.02%)
        """
        self.id = battery_id
        self.age = 0  # Age in months
        self.initial_capacity = initial_capacity
        self.capacity = initial_capacity
        self.health = 1.0
        self.status = BatteryStatus.NEW
        self.original_owner = original_owner
        self.manufacture_date = datetime.now()
        self.cycle_count = 0
        self.degradation_rate = degradation_rate
        print(f"Created new battery {battery_id} with initial capacity {initial_capacity} kWh")
        
    def degrade_battery(self, cycles: int = 1) -> None:
        """Simulate battery degradation over time and usage.
        
        Uses a more realistic non-linear degradation model:
        - Initial phase: Slow degradation (break-in period)
        - Middle phase: Linear degradation 
        - End phase: Accelerated degradation (knee point)
        
        Args:
            cycles (int): Number of charge cycles to simulate
        """
        previous_health = self.health
        
        for _ in range(cycles):
            # Non-linear degradation model
            if self.cycle_count < 200:  
                # Phase 1, Initial slow degradation
                degradation_factor = 0.7
            elif self.cycle_count > 1000:  
                # Phase 3, End-of-life accelerated degradation
                degradation_factor = 1.5
            else:
                # Phase 2, Normal linear degradation
                degradation_factor = 1.0
                

            capacity_loss = self.initial_capacity * self.degradation_rate * degradation_factor
            self.capacity -= capacity_loss
            self.cycle_count += 1
            
        #healh as percentage
        self.health = max(0.0, self.capacity / self.initial_capacity)
        

        health_drop = previous_health - self.health
        if health_drop > 0.05:  # health drop meer dan 5%
            print(f"Battery {self.id} health dropped significantly: {previous_health:.2f} -> {self.health:.2f} (age: {self.age} months, cycles: {self.cycle_count})")
        


        if self.status == BatteryStatus.IN_USE:
            if self.health < self.SECOND_LIFE_THRESHOLD:
                self.change_status(BatteryStatus.END_OF_LIFE)
                print(f"Battery {self.id} changed to END_OF_LIFE. Health: {self.health:.2f}, Age: {self.age} months, Cycles: {self.cycle_count}")
            elif self.health < self.GOOD_HEALTH_THRESHOLD and self.age > 60:  # alleen second life voor oude batterij
                print(f"Battery {self.id} now in second-life range. Health: {self.health:.2f}, Age: {self.age} months, Cycles: {self.cycle_count}")
        
    def assess_condition(self) -> dict:
        """Assess the current condition of the battery.
        
        Returns:
            dict: Assessment results including:
                - health: float (0-1)
                - status: BatteryStatus
                - age: int (months)
                - cycle_count: int
                - remaining_capacity: float (kWh)
        """
        return {
            'health': self.health,
            'status': self.status,
            'age': self.age,
            'cycle_count': self.cycle_count,
            'remaining_capacity': self.capacity
        }
    
    def update_age(self, months: int = 1) -> None:
        """Update the age of the battery.
        
        Args:
            months (int): Number of months to age the battery
        """
        self.age += months
        if self.age % 12 == 0:  # Voor elk jaar
            print(f"Battery {self.id} is now {self.age} months old. Health: {self.health:.2f}")
        
    def change_status(self, new_status: BatteryStatus) -> None:
        """Change the status of the battery.
        
        Args:
            new_status (BatteryStatus): New status to set
        """
        old_status = self.status
        self.status = new_status
        print(f"Battery {self.id} status changed: {old_status.value} -> {new_status.value}")
        
    def is_suitable_for_second_life(self) -> bool:
        """Determine if battery is suitable for second-life applications.
        
        Returns:
            bool: True if battery health is between SECOND_LIFE_THRESHOLD and GOOD_HEALTH_THRESHOLD
        """
        is_suitable = (self.SECOND_LIFE_THRESHOLD <= self.health < self.GOOD_HEALTH_THRESHOLD)
        return is_suitable 