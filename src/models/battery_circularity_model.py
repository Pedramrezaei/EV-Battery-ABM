from mesa import Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from typing import Dict

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

from src.agents.eVOwner import EVOwner
from src.agents.car_manufacturer import CarManufacturer
from src.agents.recycling_facility import RecyclingFacility
from src.agents.battery_refurbisher import BatteryRefurbisher
from src.agents.battery import Battery, BatteryStatus
from src.utils.constants import (
    DEFAULT_GRID_WIDTH,
    DEFAULT_GRID_HEIGHT,
    DEFAULT_NUM_OWNERS,
    DEFAULT_NUM_MANUFACTURERS,
    DEFAULT_NUM_RECYCLERS,
    DEFAULT_NUM_REFURBISHERS,
    INCOME_RANGE,
    ENVIRONMENTAL_CONSCIOUSNESS_RANGE,
    DEFAULT_PRODUCTION_CAPACITY,
    DEFAULT_RECYCLING_COMMITMENT,
    DEFAULT_PROCESSING_CAPACITY,
    DEFAULT_EFFICIENCY_RATE,
    DEFAULT_TECHNICAL_CAPABILITY,
    DEFAULT_REFURBISHER_CAPACITY
)

class BatteryCircularityModel(Model):
    """Model to simulate battery circularity in the Netherlands.
    
    This model coordinates the interactions between:
    - EV Owners (battery users)
    - Car Manufacturers (battery producers)
    - Recycling Facilities (end-of-life processing)
    - Battery Refurbishers (second-life applications)
    """
    
    def __init__(self,
                 num_owners: int = DEFAULT_NUM_OWNERS,
                 num_manufacturers: int = DEFAULT_NUM_MANUFACTURERS,
                 num_recyclers: int = DEFAULT_NUM_RECYCLERS,
                 num_refurbishers: int = DEFAULT_NUM_REFURBISHERS,
                 width: int = DEFAULT_GRID_WIDTH,
                 height: int = DEFAULT_GRID_HEIGHT):
        """Initialize the battery circularity model.
        
        Args:
            num_owners (int): Number of EV owners
            num_manufacturers (int): Number of car manufacturers
            num_recyclers (int): Number of recycling facilities
            num_refurbishers (int): Number of battery refurbishers
            width (int): Grid width
            height (int): Grid height
        """
        super().__init__()
        self.num_owners = num_owners
        self.num_manufacturers = num_manufacturers
        self.num_recyclers = num_recyclers
        self.num_refurbishers = num_refurbishers
        
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        
        # Lists to keep track of different agent types
        self.manufacturers = []
        self.recyclers = []
        self.refurbishers = []
        self.owners = []
        
        # NEW: Track all batteries in the system, including those not with owners
        self.all_batteries = []
        self.stranded_batteries = []  # Batteries that couldn't be handed off properly
        
        # Create agents
        self.create_manufacturers()
        self.create_recyclers()
        self.create_refurbishers()
        self.create_owners()
        
        # Initial battery distribution
        self.distribute_initial_batteries()
        
        # Set up data collection
        self.datacollector = DataCollector(
            model_reporters={
                # Battery lifecycle metrics
                "Active_Batteries": lambda m: self.count_batteries_by_status(BatteryStatus.IN_USE),
                "End_Of_Life": lambda m: self.count_batteries_by_status(BatteryStatus.END_OF_LIFE),
                "Collected": lambda m: self.count_batteries_by_status(BatteryStatus.COLLECTED),
                "Recycled": lambda m: self.count_batteries_by_status(BatteryStatus.RECYCLED),
                "Refurbished": lambda m: self.count_batteries_by_status(BatteryStatus.REFURBISHED),
                "Stranded": lambda m: len(self.stranded_batteries),
                
                # Circularity metrics
                "Recycling_Rate": lambda m: self.calculate_recycling_rate(),
                "Second_Life_Rate": lambda m: self.calculate_second_life_rate(),
                "Total_Grid_Storage": lambda m: self.calculate_total_grid_storage(),
                
                # Material recovery metrics
                "Materials_Recovered": lambda m: self.get_total_materials_recovered(),
                
                # Economic metrics
                "Average_Battery_Age": lambda m: self.calculate_average_battery_age()
            }
        )
        
    def distribute_initial_batteries(self):
        """Distribute initial batteries to a portion of EV owners with realistic age distribution."""
        print("Distributing initial batteries with age distribution...")
        
        # Give batteries to most of the owners (80%)
        for owner in self.owners[:int(self.num_owners * 0.8)]:
            manufacturer = self.random.choice(self.manufacturers)
            battery = manufacturer.produce_battery(owner)
            
            if battery:
                # Assign a random initial age between 0 and 60 months (0-5 years)
                initial_age = self.random.randint(0, 60)
                
                # Apply aging and degradation to simulate real-world usage
                battery.status = BatteryStatus.IN_USE
                
                # Age the battery to its initial age
                for _ in range(initial_age):
                    battery.update_age(1)
                    # Add some random degradation for each month
                    cycles = self.random.randint(10, 20)
                    battery.degrade_battery(cycles=cycles)
                
                # Set owner's battery and ownership duration
                owner.battery = battery
                owner.ownership_duration = initial_age
                
                # Add to tracked batteries
                self.all_batteries.append(battery)
                
                print(f"Owner {owner.unique_id} received battery {battery.id} " +
                      f"with initial age {initial_age} months and health {battery.health:.2f}")
                
    def track_battery(self, battery):
        """Add a battery to the tracking system if it's not already tracked."""
        if battery not in self.all_batteries:
            self.all_batteries.append(battery)
            
    def track_stranded_battery(self, battery):
        """Track a battery that couldn't be properly handed off."""
        if battery not in self.stranded_batteries:
            self.stranded_batteries.append(battery)
            print(f"Battery {battery.id} is now stranded in the system")
            
    def create_manufacturers(self) -> None:
        """Create car manufacturer agents."""
        for i in range(self.num_manufacturers):
            manufacturer = CarManufacturer(
                f"Manufacturer_{i}",
                self,
                production_capacity=DEFAULT_PRODUCTION_CAPACITY,
                recycling_commitment=DEFAULT_RECYCLING_COMMITMENT
            )
            self.schedule.add(manufacturer)
            self.manufacturers.append(manufacturer)
            # Random grid placement
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(manufacturer, (x, y))
            
    def create_recyclers(self) -> None:
        """Create recycling facility agents."""
        for i in range(self.num_recyclers):
            recycler = RecyclingFacility(
                f"Recycler_{i}",
                self,
                processing_capacity=DEFAULT_PROCESSING_CAPACITY,
                efficiency_rate=DEFAULT_EFFICIENCY_RATE
            )
            self.schedule.add(recycler)
            self.recyclers.append(recycler)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(recycler, (x, y))
            
    def create_refurbishers(self) -> None:
        """Create battery refurbisher agents."""
        for i in range(self.num_refurbishers):
            refurbisher = BatteryRefurbisher(
                f"Refurbisher_{i}",
                self,
                technical_capability=DEFAULT_TECHNICAL_CAPABILITY,
                capacity=DEFAULT_REFURBISHER_CAPACITY
            )
            self.schedule.add(refurbisher)
            self.refurbishers.append(refurbisher)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(refurbisher, (x, y))
            
    def create_owners(self) -> None:
        """Create EV owner agents."""
        for i in range(self.num_owners):
            # Random income between 30k and 120k
            income = self.random.uniform(INCOME_RANGE[0], INCOME_RANGE[1])
            # Random environmental consciousness
            env_consciousness = self.random.uniform(ENVIRONMENTAL_CONSCIOUSNESS_RANGE[0], ENVIRONMENTAL_CONSCIOUSNESS_RANGE[1])
            
            owner = EVOwner(
                f"Owner_{i}",
                self,
                income=income,
                environmental_consciousness=env_consciousness
            )
            self.schedule.add(owner)
            self.owners.append(owner)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(owner, (x, y))
            
    def handle_end_of_life_batteries(self):
        """Process end-of-life batteries."""
        eol_count = sum(1 for owner in self.owners if owner.battery and owner.battery.status == BatteryStatus.END_OF_LIFE)
        if eol_count > 0:
            print(f"Processing {eol_count} end-of-life batteries")
            
        # Track the success rate of handoffs
        successful_handoffs = 0
        
        for owner in self.owners:
            if owner.battery and owner.battery.status == BatteryStatus.END_OF_LIFE:
                decision = owner.make_end_of_life_decision()
                
                if decision in ['recycle', 'refurbish']:
                    # Find a manufacturer to take back the battery
                    for manufacturer in self.manufacturers:
                        old_battery = owner.battery
                        success = manufacturer.handle_take_back(old_battery)
                        if success:
                            successful_handoffs += 1
                            print(f"Battery from owner {owner.unique_id} successfully taken back by manufacturer {manufacturer.unique_id}")
                            # Important: Only set owner's battery to None AFTER successful handoff
                            owner.battery = None
                            owner.ownership_duration = 0
                            break
                    else:  # This runs if no break occurred in the for loop
                        print(f"WARNING: No manufacturer took back the battery from owner {owner.unique_id}")
                        # Track this stranded battery
                        self.track_stranded_battery(owner.battery)
                        owner.battery = None
                        owner.ownership_duration = 0
                elif decision == 'keep':
                    print(f"Owner {self.unique_id} decided to keep their end-of-life battery")
        
        # Report on handoff success
        if eol_count > 0:
            print(f"End-of-life battery handoff success rate: {successful_handoffs}/{eol_count} " +
                  f"({(successful_handoffs/eol_count)*100:.1f}%)")
                        
    def forward_to_refurbisher(self, battery: Battery) -> bool:
        """Forward a battery to a refurbisher.
        
        Args:
            battery (Battery): Battery to be refurbished
            
        Returns:
            bool: True if a refurbisher accepted the battery
        """
        if not self.refurbishers:
            print("WARNING: No refurbishers available in the model")
            return False
            
        # Ensure the battery is being tracked
        self.track_battery(battery)
            
        # Try all refurbishers until one accepts the battery
        for refurbisher in self.refurbishers:
            if refurbisher.receive_battery(battery):
                print(f"Refurbisher {refurbisher.unique_id} accepted battery {battery.id}")
                return True
                
        print(f"WARNING: No refurbisher had capacity for battery {battery.id}")
        # If no refurbisher took it, track it as stranded
        self.track_stranded_battery(battery)
        return False
                
    def forward_to_recycler(self, battery: Battery) -> bool:
        """Forward a battery to a recycler.
        
        Args:
            battery (Battery): Battery to be recycled
            
        Returns:
            bool: True if a recycler accepted the battery
        """
        if not self.recyclers:
            print("WARNING: No recyclers available in the model")
            return False
            
        # Ensure the battery is being tracked
        self.track_battery(battery)
            
        # Try all recyclers until one accepts the battery
        for recycler in self.recyclers:
            if recycler.receive_battery(battery):
                print(f"Recycler {recycler.unique_id} accepted battery {battery.id}")
                return True
                
        print(f"WARNING: No recycler had capacity for battery {battery.id}")
        # If no recycler took it, track it as stranded
        self.track_stranded_battery(battery)
        return False
                
    def handle_battery_purchases(self):
        """Process new battery purchases."""
        purchase_requests = sum(1 for owner in self.owners if owner.make_purchase_decision())
        if purchase_requests > 0:
            print(f"Processing {purchase_requests} battery purchase requests")
            
        for owner in self.owners:
            if owner.make_purchase_decision():
                # Try all manufacturers until one succeeds
                for manufacturer in self.manufacturers:
                    new_battery = manufacturer.produce_battery(owner)
                    if new_battery:
                        owner.battery = new_battery
                        new_battery.status = BatteryStatus.IN_USE
                        print(f"Owner {owner.unique_id} received new battery {new_battery.id} from manufacturer {manufacturer.unique_id}")
                        break
                else:  # This runs if no break occurred in the for loop
                    print(f"WARNING: No manufacturer could produce a battery for owner {owner.unique_id}")
                    
    def step(self) -> None:
        """Execute one step of the model."""
        print(f"\n--- STEP {self.schedule.steps + 1} ---")
        
        # Update all agents
        self.schedule.step()
        
        # Get battery counts before lifecycle handling
        before_counts = self.get_all_battery_counts()
        print("Battery counts before lifecycle handling:")
        for status, count in before_counts.items():
            print(f"  {status}: {count}")
        
        # Handle battery lifecycle events
        self.handle_end_of_life_batteries()
        self.handle_battery_purchases()
        
        # Process batteries at facilities
        print("\nProcessing facilities:")
        for recycler in self.recyclers:
            print(f"Recycler {recycler.unique_id} has {len(recycler.current_inventory)} batteries in queue")
            recycler.step()
            
        for refurbisher in self.refurbishers:
            print(f"Refurbisher {refurbisher.unique_id} has {len(refurbisher.inventory)} batteries in queue")
            refurbisher.step()
            
        # Update model data
        self.datacollector.collect(self)
        
        # Get battery counts after all processing
        after_counts = self.get_all_battery_counts()
        print("\nBattery counts after all processing:")
        for status, count in after_counts.items():
            print(f"  {status}: {count}")
        print("-------------------")
        
    def count_batteries_by_status(self, status: BatteryStatus) -> int:
        """Count batteries in a particular status across the entire system.
        
        This method counts batteries in all possible locations:
        - With owners
        - In recycler inventory
        - In refurbisher inventory
        - In stranded inventory
        - Throughout the system
        
        Args:
            status (BatteryStatus): The battery status to count
            
        Returns:
            int: The number of batteries with the given status
        """
        count = 0
        
        # Count batteries with owners
        for owner in self.owners:
            if owner.battery and owner.battery.status == status:
                count += 1
                
        # Count batteries in recycler inventory
        for recycler in self.recyclers:
            for battery in recycler.current_inventory:
                if battery.status == status:
                    count += 1
                    
        # Count batteries in refurbisher inventory
        for refurbisher in self.refurbishers:
            for battery in refurbisher.inventory:
                if battery.status == status:
                    count += 1
                    
        # Count stranded batteries
        for battery in self.stranded_batteries:
            if battery.status == status:
                count += 1
                
        # For RECYCLED and REFURBISHED status, we need to count all batteries
        # that have ever reached these states, since they're terminal states
        if status == BatteryStatus.RECYCLED:
            count = sum(recycler.total_processed for recycler in self.recyclers)
        elif status == BatteryStatus.REFURBISHED:
            count = sum(refurbisher.successful_conversions for refurbisher in self.refurbishers)
            
        return count
        
    def calculate_recycling_rate(self) -> float:
        """Calculate the current recycling rate."""
        total_end_of_life = sum(1 for agent in self.schedule.agents 
                               if isinstance(agent, EVOwner) and agent.battery 
                               and agent.battery.status == BatteryStatus.END_OF_LIFE)
        total_recycled = sum(1 for agent in self.schedule.agents 
                           if isinstance(agent, EVOwner) and agent.battery 
                           and agent.battery.status == BatteryStatus.RECYCLED)
        
        if total_end_of_life == 0:
            return 0.0
        return total_recycled / total_end_of_life
        
    def calculate_second_life_rate(self) -> float:
        """Calculate the rate of batteries going to second life.
        
        Returns:
            float: Proportion of end-of-life batteries that get refurbished
                  instead of recycled
        """
        # Count total batteries that have been processed by both types of facilities
        total_recycled = sum(recycler.total_processed for recycler in self.recyclers)
        total_refurbished = sum(refurbisher.successful_conversions for refurbisher in self.refurbishers)
        
        total_processed = total_recycled + total_refurbished
        
        if total_processed == 0:
            return 0.0
        
        # Second life rate is the proportion of processed batteries that were refurbished
        return total_refurbished / total_processed
        
    def calculate_total_grid_storage(self) -> float:
        """Calculate total grid storage capacity from refurbished batteries."""
        return sum(refurbisher.convert_to_grid_storage(refurbisher.inventory)
                  for refurbisher in self.refurbishers)
        
    def get_total_materials_recovered(self) -> Dict[str, float]:
        """Get total materials recovered across all recycling facilities.
        
        Returns:
            Dict[str, float]: Amount of each material recovered in kg
        """
        totals = {
            'lithium': 0.0,
            'cobalt': 0.0,
            'nickel': 0.0,
            'copper': 0.0
        }
        
        # Sum materials recovered across all recycling facilities
        for recycler in self.recyclers:
            for material, amount in recycler.total_materials.items():
                totals[material] += amount
                
        return totals
        
    def calculate_average_battery_age(self) -> float:
        """Calculate average age of active batteries."""
        ages = [agent.battery.age for agent in self.schedule.agents 
                if isinstance(agent, EVOwner) and agent.battery]
        
        if not ages:
            return 0.0
        return sum(ages) / len(ages)
        
    def get_recycling_statistics(self) -> Dict:
        """Get comprehensive statistics about the recycling process.
        
        Returns:
            Dict containing:
                - total_processed: Total batteries processed by recyclers
                - avg_efficiency: Average recovery efficiency
                - materials_by_type: Materials recovered by type
                - queue_length: Current number of batteries waiting
        """
        total_processed = sum(recycler.total_processed for recycler in self.recyclers)
        total_in_queue = sum(len(recycler.current_inventory) for recycler in self.recyclers)
        
        # Calculate average efficiency if any batteries have been processed
        if total_processed > 0:
            avg_efficiency = sum(recycler.efficiency_rate * recycler.total_processed 
                               for recycler in self.recyclers) / total_processed
        else:
            avg_efficiency = 0.0
            
        return {
            'total_processed': total_processed,
            'avg_efficiency': avg_efficiency,
            'materials_by_type': self.get_total_materials_recovered(),
            'queue_length': total_in_queue
        }
    
    def get_refurbishment_statistics(self) -> Dict:
        """Get comprehensive statistics about the refurbishment process.
        
        Returns:
            Dict containing:
                - total_refurbished: Total batteries successfully refurbished
                - avg_success_rate: Average success rate of refurbishment
                - grid_storage: Total storage capacity created
                - queue_length: Current number of batteries waiting
        """
        total_refurbished = sum(refurbisher.successful_conversions for refurbisher in self.refurbishers)
        total_in_queue = sum(len(refurbisher.inventory) for refurbisher in self.refurbishers)
        
        # Calculate average success rate if any refurbishers exist
        if self.refurbishers:
            avg_capability = sum(refurbisher.technical_capability for refurbisher in self.refurbishers) / len(self.refurbishers)
        else:
            avg_capability = 0.0
            
        return {
            'total_refurbished': total_refurbished,
            'avg_success_rate': avg_capability,
            'grid_storage': self.calculate_total_grid_storage(),
            'queue_length': total_in_queue
        }
    
    def get_all_battery_counts(self) -> Dict[str, int]:
        """Get a comprehensive count of all batteries in the system by status.
        
        This includes batteries:
        - With owners (in use, end of life)
        - In recycling facility queues
        - In refurbisher queues
        - Already processed (recycled, refurbished)
        
        Returns:
            Dict mapping status descriptions to counts
        """
        # Count batteries with owners
        owner_batteries = {
            'In Use': self.count_batteries_by_status(BatteryStatus.IN_USE),
            'End of Life': self.count_batteries_by_status(BatteryStatus.END_OF_LIFE),
        }
        
        # Count batteries in processing queues
        in_processing = {
            'In Recycler Queue': sum(len(recycler.current_inventory) for recycler in self.recyclers),
            'In Refurbisher Queue': sum(len(refurbisher.inventory) for refurbisher in self.refurbishers),
        }
        
        # Count processed batteries
        processed = {
            'Recycled': sum(recycler.total_processed for recycler in self.recyclers),
            'Refurbished': sum(refurbisher.successful_conversions for refurbisher in self.refurbishers),
        }
        
        # Combine all counts
        result = {}
        result.update(owner_batteries)
        result.update(in_processing)
        result.update(processed)
        
        return result 