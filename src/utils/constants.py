"""
Constants for EV Battery Circularity ABM

This file contains all constants used throughout the model. Each constant should include a reference to its source
if available and a brief explanation of its meaning.
"""

# Simulation parameters
# ---------------------
DEFAULT_GRID_WIDTH = 50  # Width of simulation grid
DEFAULT_GRID_HEIGHT = 50  # Height of simulation grid
DEFAULT_SIMULATION_STEPS = 50  # Default number of steps to run the simulation
DEFAULT_TIME_STEP_MONTHS = 1  # Each step represents 1 month of time


# Agent population sizes
# ---------------------
DEFAULT_NUM_OWNERS = 100  # Default number of EV owners in the simulation
DEFAULT_NUM_MANUFACTURERS = 3  # Default number of car manufacturers
DEFAULT_NUM_RECYCLERS = 2  # Default number of recycling facilities
DEFAULT_NUM_REFURBISHERS = 2  # Default number of battery refurbishers


# Battery parameters
# -----------------
# Health thresholds
BATTERY_GOOD_HEALTH_THRESHOLD = 0.8  # Above this threshold, battery is in good condition
BATTERY_SECOND_LIFE_THRESHOLD = 0.6  # Above this threshold, battery is suitable for second life

# Battery degradation
# Typical EV batteries lose 2-3% capacity per year, which is about 0.0005-0.0008 per cycle
# with 200-300 cycles per year (Source: Various battery studies)
DEFAULT_DEGRADATION_RATE = 0.0002  # Reduced from 0.0005 - more realistic degradation of 0.02% per cycle
DEFAULT_BATTERY_CAPACITY = 75.0  # Default battery capacity in kWh (typical EV battery)
DEFAULT_BATTERY_CYCLES_PER_MONTH = (10, 20)  # Increased from (1, 3) - typical EVs do 10-20 cycles per month

# Material content (kg) for a typical EV battery
BATTERY_MATERIALS = {
    'lithium': 10.0,
    'cobalt': 30.0,
    'nickel': 40.0,
    'copper': 25.0
}


# Manufacturer parameters
# ----------------------
DEFAULT_PRODUCTION_CAPACITY = 3  # Number of batteries a manufacturer can produce per step
DEFAULT_RECYCLING_COMMITMENT = 0.7  # Probability of accepting end-of-life batteries
DEFAULT_WARRANTY_AGE_LIMIT = 96  # Default warranty period in months (8 years)
DEFAULT_WARRANTY_HEALTH_THRESHOLD = 0.7  # Minimum health level covered by warranty


# Recycling facility parameters
# ---------------------------
DEFAULT_PROCESSING_CAPACITY = 5  # Number of batteries that can be processed per step
DEFAULT_EFFICIENCY_RATE = 0.8  # Efficiency of material recovery (0-1)


# Battery refurbisher parameters
# ----------------------------
DEFAULT_TECHNICAL_CAPABILITY = 0.75  # Technical expertise level (0-1)
DEFAULT_REFURBISHER_CAPACITY = 3  # Number of batteries that can be processed per step
DEFAULT_GRID_STORAGE_FACTOR = (0.7, 0.9)  # Range of usable capacity for grid storage


# EV Owner parameters
# ------------------
INCOME_RANGE = (30000, 120000)  # Range of annual income for EV owners
ENVIRONMENTAL_CONSCIOUSNESS_RANGE = (0.2, 0.9)  # Range of environmental awareness values

# Purchase decision parameters
NO_BATTERY_PURCHASE_PROBABILITY = 0.8  # Probability of purchasing when no battery
MIN_OWNERSHIP_FOR_REPLACEMENT = 72  # Increased from 24 to 72 months (6 years) - more realistic minimum ownership
END_OF_LIFE_PURCHASE_PROBABILITY = 0.8  # Increased from 0.6 to 0.8 - higher probability when battery is at end of life
BASE_PURCHASE_PROBABILITY = 0.01  # Reduced from 0.05 to 0.01 - much lower base probability for functioning batteries

# End-of-life decision parameters
REFURBISH_PROBABILITY_FACTOR = 0.7  # Increased from 0.5 to 0.7 - encourage more refurbishment 