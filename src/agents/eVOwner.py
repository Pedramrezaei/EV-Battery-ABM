import random
from typing import Optional, TYPE_CHECKING

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.agents.agent import Agent
from src.agents.battery import BatteryStatus, Battery
from src.utils.constants import (
    MIN_OWNERSHIP_FOR_REPLACEMENT,
    NO_BATTERY_PURCHASE_PROBABILITY,
    END_OF_LIFE_PURCHASE_PROBABILITY,
    BASE_PURCHASE_PROBABILITY,
    REFURBISH_PROBABILITY_FACTOR,
    DEFAULT_BATTERY_CYCLES_PER_MONTH
)

if TYPE_CHECKING:
    from src.agents.battery import Battery

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
        self.battery: Optional[Battery] = None
        self.ownership_duration = 0
        
        # new attributes for EOL handling
        self.has_eol_decision = False
        self.eol_decision = None
        self.eol_waiting_time = 0
        self.eol_waiting_duration = 0
        
    def make_purchase_decision(self) -> bool:
        """Decide whether to purchase a new battery.
        
        The decision is based on:
        - Current battery age and condition (if any)
        - Income level
        - Environmental consciousness
        
        Returns:
            bool: True if decided to purchase, False otherwise
        """
        # no battery, high chance of purchase
        if self.battery is None:
            return random.random() < NO_BATTERY_PURCHASE_PROBABILITY

        if self.ownership_duration % 12 != 0:
            return False
            
        # minimum ownership duration
        if self.ownership_duration < MIN_OWNERSHIP_FOR_REPLACEMENT:
            return False
            

        battery_condition = self.battery.assess_condition()
        
        # very old batteries, increase purchase probability
        age_factor = min(0.7, self.ownership_duration / 120)

        if battery_condition['health'] < Battery.SECOND_LIFE_THRESHOLD:
            purchase_probability = END_OF_LIFE_PURCHASE_PROBABILITY
            print(f"Owner {self.unique_id} considering purchase: Critical battery health {battery_condition['health']:.2f}")
        elif battery_condition['health'] < Battery.GOOD_HEALTH_THRESHOLD:

            purchase_probability = END_OF_LIFE_PURCHASE_PROBABILITY * 0.5 * (1 - (battery_condition['health'] - Battery.SECOND_LIFE_THRESHOLD) / 
                                                                      (Battery.GOOD_HEALTH_THRESHOLD - Battery.SECOND_LIFE_THRESHOLD))
            print(f"Owner {self.unique_id} considering purchase: Poor battery health {battery_condition['health']:.2f}")
        else:

            purchase_probability = (
                BASE_PURCHASE_PROBABILITY +
                0.05 * (self.income / 100000) +
                0.02 * self.environmental_consciousness +
                age_factor
            )
            
        # maximum probability
        purchase_probability = min(purchase_probability, 0.9)
        
        decision = random.random() < purchase_probability
        if decision:
            print(f"Owner {self.unique_id} decided to purchase new battery after {self.ownership_duration} months. " +
                  f"Health: {battery_condition['health']:.2f}, Age: {battery_condition['age']} months")
        
        return decision
        
    def handle_battery_replacement(self) -> None:
        """Properly handle the old battery when purchasing a new one.
        
        This ensures the old battery is properly sent for recycling/refurbishment
        and doesn't disappear from the system.
        """
        if not self.battery:
            return
            

        if self.battery.status != BatteryStatus.END_OF_LIFE:
            self.battery.status = BatteryStatus.END_OF_LIFE
            print(f"Battery {self.battery.id} marked as END_OF_LIFE for replacement")
        
        # hand off the battery to a manufacturer
        old_battery = self.battery
        handed_off = False
        
        for manufacturer in self.model.manufacturers:
            if manufacturer.handle_take_back(old_battery):
                print(f"Owner {self.unique_id} handed off battery {old_battery.id} to manufacturer {manufacturer.unique_id}")
                handed_off = True
                break
                
        if not handed_off:
            print(f"WARNING: Owner {self.unique_id} couldn't find a manufacturer to take battery {old_battery.id}")
            #  no manufacturer takes it -> stranded
            self.model.track_stranded_battery(old_battery)
            
        # reset properties for the new battery
        self.battery = None
        self.ownership_duration = 0
        self.has_eol_decision = False
        self.eol_decision = None
        self.eol_waiting_time = 0
        self.eol_waiting_duration = 0
        
    def make_end_of_life_decision(self) -> str:
        """Decide what to do with the battery at end of life.
        
        Also sets a waiting period before the battery is handed off.
        
        Returns:
            str: Decision ('recycle', 'refurbish', or 'keep')
        """
        if not self.battery:
            return 'none'
            
        if self.battery.health >= Battery.GOOD_HEALTH_THRESHOLD:
            return 'keep'

        if self.battery.is_suitable_for_second_life():
            # Higher environmental consciousness increases chance of refurbishment
            refurb_chance = REFURBISH_PROBABILITY_FACTOR + REFURBISH_PROBABILITY_FACTOR * self.environmental_consciousness
            decision = 'refurbish' if random.random() < refurb_chance else 'recycle'
            print(f"Owner {self.unique_id} with env_con {self.environmental_consciousness:.2f} decided: {decision} for battery with health {self.battery.health:.2f}")

            self.has_eol_decision = True
            self.eol_decision = decision
            self.eol_waiting_time = 0
            self.eol_waiting_duration = random.randint(1, 3)
            print(f"Owner {self.unique_id} will wait {self.eol_waiting_duration} months before handing off the battery")
            
            return decision
                
        # Default to recycling if bat health is too low
        decision = 'recycle'
        print(f"Owner {self.unique_id} decided: recycle for battery with health {self.battery.health:.2f} (too low for refurb)")
        
        # waiting period  1-3 months
        self.has_eol_decision = True
        self.eol_decision = decision
        self.eol_waiting_time = 0
        self.eol_waiting_duration = random.randint(1, 3)
        print(f"Owner {self.unique_id} will wait {self.eol_waiting_duration} months before handing off the battery")
        
        return decision
        
    def maintain_battery(self) -> None:
        """Perform maintenance check on the battery."""
        if not self.battery:
            return
            

        self.battery.update_age(months=1)
        self.ownership_duration += 1
        

        cycles = self.random.randint(DEFAULT_BATTERY_CYCLES_PER_MONTH[0], DEFAULT_BATTERY_CYCLES_PER_MONTH[1])
        self.battery.degrade_battery(cycles=cycles)

        if (self.battery.status == BatteryStatus.IN_USE and 
            self.battery.health < Battery.SECOND_LIFE_THRESHOLD):
            self.battery.status = BatteryStatus.END_OF_LIFE
            print(f"Battery of owner {self.unique_id} reached END_OF_LIFE. Age: {self.battery.age} months, Health: {self.battery.health:.2f}")
            
    def update_ownership_duration(self) -> None:
        """Update the duration of battery ownership."""
        pass
        
    def step(self) -> None:
        """Execute one step of the EV Owner agent."""

        self.maintain_battery()

        if (self.battery and 
            self.battery.status == BatteryStatus.END_OF_LIFE and 
            self.has_eol_decision and 
            self.eol_decision in ['recycle', 'refurbish']):
            
            self.eol_waiting_time += 1
            
            if self.eol_waiting_time >= self.eol_waiting_duration:
                print(f"Owner {self.unique_id} finished waiting period for end-of-life battery {self.battery.id}")
                self.handle_battery_replacement()

                return
        
        # consider purchasing new bat
        if not (self.battery and self.battery.status == BatteryStatus.END_OF_LIFE and self.has_eol_decision):
            purchase_decision = self.make_purchase_decision()
            if purchase_decision:
                print(f"Owner {self.unique_id} wants to purchase a new battery")
                # IMPORTANT: handle the old bat properly before getting a new one
                if self.battery:
                    self.handle_battery_replacement()
            
        # check if EOL decision needed
        if (not self.has_eol_decision and 
            self.battery and 
            self.battery.status == BatteryStatus.END_OF_LIFE):
            decision = self.make_end_of_life_decision()
            print(f"Owner {self.unique_id} made end-of-life decision: {decision}")