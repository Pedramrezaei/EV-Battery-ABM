
from enum import Enum
from datetime import datetime
from .eVOwner import EVOwner

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
        manufacture_date (datetime): When the battery was manufactured
        initial_capacity (float): The original capacity when new
        cycle_count (int): Number of charge cycles
        degradation_rate (float): Rate at which battery degrades per cycle
        
    Health Thresholds:
        1.0 - 0.8: Good condition (NEW/IN_USE)
        0.8 - 0.6: End of first life, suitable for second life
        < 0.6: End of life, needs recycling
    """
    
    # Health thresholds
    GOOD_HEALTH_THRESHOLD = 0.8
    SECOND_LIFE_THRESHOLD = 0.6
    
    def __init__(self, 
                 battery_id: str, 
                 initial_capacity: float,
                 original_owner: 'EVOwner',
                 degradation_rate: float = 0.0005) -> None:
        """Initialize a new battery.
        
        Args:
            battery_id (str): Unique identifier for the battery
            initial_capacity (float): Initial capacity in kWh
            original_owner (EVOwner): Original owner of the battery
            degradation_rate (float): Rate of capacity loss per cycle (default 0.05%)
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
        
    def degrade_battery(self, cycles: int = 1) -> None:
        """Simulate battery degradation over time and usage.
        
        Args:
            cycles (int): Number of charge cycles to simulate
        """
        for _ in range(cycles):
            # Linear degradation model (can be made more sophisticated)
            capacity_loss = self.initial_capacity * self.degradation_rate
            self.capacity -= capacity_loss
            self.cycle_count += 1
            
        # Update health as percentage of initial capacity
        self.health = max(0.0, self.capacity / self.initial_capacity)
        
        # Update status based on health thresholds
        if self.status == BatteryStatus.IN_USE:
            if self.health < self.SECOND_LIFE_THRESHOLD:
                self.status = BatteryStatus.END_OF_LIFE
            elif self.health < self.GOOD_HEALTH_THRESHOLD:
                # Battery is in range for second life consideration
                pass  # Status will be changed when decision is made
        
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
        
    def change_status(self, new_status: BatteryStatus) -> None:
        """Change the status of the battery.
        
        Args:
            new_status (BatteryStatus): New status to set
        """
        self.status = new_status
        
    def is_suitable_for_second_life(self) -> bool:
        """Determine if battery is suitable for second-life applications.
        
        Returns:
            bool: True if battery health is between SECOND_LIFE_THRESHOLD and GOOD_HEALTH_THRESHOLD
        """
        return self.SECOND_LIFE_THRESHOLD <= self.health < self.GOOD_HEALTH_THRESHOLD 