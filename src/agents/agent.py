from mesa import Agent as MesaAgent
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class Location:
    """Represents a location in the simulation space. """

    x: float
    y: float

class Agent(MesaAgent):
    """Base Super agent class that all specific agents inherit from
       This agent class is used to create all agents in the simulation.
    
    Attributes:
        id (str): Unique identifier for the agent
        location (Location): Current location of the agent
        model: The model instance this agent is part of
    """
    
    def __init__(self, unique_id: str, model, x: float = 0.0, y: float = 0.0) -> None:
        """Initialize a new agent.
        
        Args:
            unique_id (str): Unique identifier for the agent
            model: The model instance this agent is part of
            x (float): Initial x coordinate. Defaults to 0.0
            y (float): Initial y coordinate. Defaults to 0.0
        """
        super().__init__(unique_id, model)
        self.location = Location(x, y)
    
    def get_position(self) -> Tuple[float, float]:
        """Get the current position of the agent.
        
        Returns:
            Tuple[float, float]: Current (x, y) coordinates
        """
        return (self.location.x, self.location.y)
    
    def set_position(self, x: float, y: float) -> None:
        """Set the position of the agent.
        
        Args:
            x (float): New x coordinate
            y (float): New y coordinate
        """
        self.location = Location(x, y)
    
    def move_to(self, target_location: Location) -> None:
        """Move the agent to a target location.
        
        Args:
            target_location (Location): The location to move to
        """
        self.location = target_location
    
    def distance_to(self, other_agent: 'Agent') -> float:
        """Calculate Euclidean distance to another agent.
        
        Args:
            other_agent (Agent): The agent to calculate distance to
            
        Returns:
            float: Euclidean distance between this agent and the other agent
        """
        dx = self.location.x - other_agent.location.x
        dy = self.location.y - other_agent.location.y
        return (dx * dx + dy * dy) ** 0.5
    
    def step(self) -> None:
        """Take one step for each agent.

        This method will not be implemented in the super class.
        It will be implemented by the child classes as each agent will have different behavior.
        """
        pass 