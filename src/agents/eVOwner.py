from .agent import Agent
import random
from .battery import BatteryStatus, Battery

class EVOwner(Agent):
    """Electric Vehicle Owner agent.
    
    Attributes:
        income (float): Annual income of the owner
        environmental_consciousness (float): Level of environmental awareness (0-1)
        battery (Battery): The battery owned by this agent
        ownership_duration (int): How long they've owned their current battery (months)
    """
    
    def __init__(self, 
                 unique_id: str, 
                 model, 
                 income: float,
                 environmental_consciousness: float,
                 x: float = 0.0, 
                 y: float = 0.0) -> None:
        """Initialize a new EV Owner class.
        
        Args:
            unique_id (str): Unique identifier for the agent
            model: The model instance this agent is part of
            income (float): Annual income of the owner
            environmental_consciousness (float): Level of environmental awareness (0-1)
            x (float): Initial x coordinate. Defaults to 0.0
            y (float): Initial y coordinate. Defaults to 0.0
        """
        super().__init__(unique_id, model, x, y)
        self.income = income
        self.environmental_consciousness = environmental_consciousness
        self.battery = None
        self.ownership_duration = 0
        
    def make_purchase_decision(self) -> bool:
        """Decide whether to purchase a new battery.
        
        The decision is based on:
        - Current battery age and condition (if any)
        - Income level
        - Environmental consciousness
        
        Returns:
            bool: True if decided to purchase, False otherwise
        """
        if self.battery is None:
            # No current battery, high chance of purchase
            return random.random() < 0.8
            
        # Consider battery age
        if self.ownership_duration < 24:  # Less than 2 years old
            return False
            
        # Check battery health
        battery_condition = self.battery.assess_condition()
        if battery_condition['health'] < Battery.GOOD_HEALTH_THRESHOLD:
            # Higher chance of replacement when battery is at end of first life
            purchase_probability = 0.6
        else:
            # Base probability on income and environmental consciousness
            purchase_probability = (
                0.05 +  # base probability
                0.3 * (self.income / 100000) +  # income factor
                0.2 * self.environmental_consciousness  # environmental factor
            )
            
        return random.random() < min(purchase_probability, 0.9)
        
    def make_end_of_life_decision(self) -> str:
        """Decide what to do with the battery at end of life.
        
        Returns:
            str: Decision ('recycle', 'refurbish', or 'keep')
        """
        if not self.battery:
            return 'none'
            
        # If battery is still good, keep it
        if self.battery.health >= Battery.GOOD_HEALTH_THRESHOLD:
            return 'keep'
            
        # If battery is suitable for second life, consider refurbishment
        if self.battery.is_suitable_for_second_life():
            # Higher environmental consciousness increases chance of refurbishment
            if random.random() < (0.5 + 0.5 * self.environmental_consciousness):
                return 'refurbish'
                
        # Default to recycling if health is too low
        return 'recycle'
        
    def maintain_battery(self) -> None:
        """Perform maintenance check on the battery."""
        if not self.battery:
            return
            
        # Check battery condition
        battery_condition = self.battery.assess_condition()
        if battery_condition['health'] < Battery.SECOND_LIFE_THRESHOLD:
            # Simulate wear
            self.battery.degrade_battery()
            
    def update_ownership_duration(self) -> None:
        """Update the duration of battery ownership."""
        if self.battery:
            self.ownership_duration += 1
            
    def step(self) -> None:
        """Execute one step of the EV Owner agent.
        
        This method is called every time step and coordinates the agent's actions:
        1. Update ownership duration
        2. Maintain battery if needed
        3. Make purchase decision if appropriate
        4. Make end-of-life decision if battery is degraded
        """
        self.update_ownership_duration()
        self.maintain_battery()
        
        # Consider purchasing new battery
        if self.make_purchase_decision():
            # Logic for actual purchase will be implemented when 
            # CarManufacturer agent is available
            pass
            
        # Check if end-of-life decision needed
        if self.battery and self.battery.status == BatteryStatus.END_OF_LIFE:
            decision = self.make_end_of_life_decision()
            # Logic for handling the decision will be implemented when
            # RecyclingFacility and BatteryRefurbisher agents are available