# Model Constants

## Overview

The `constants.py` file contains all the numeric constants and parameters used in the EV Battery Circularity Agent-Based Model. This file centralizes all constants in one place to:

1. Make it easier to find and modify model parameters
2. Ensure consistency across the model
3. Document the meaning and source of each parameter
4. Support sensitivity analysis and parameter tuning

## How to Use

### Importing Constants

To use constants in your code:

```python
from src.utils.constants import (
    # List specific constants you need
    DEFAULT_BATTERY_CAPACITY,
    BATTERY_GOOD_HEALTH_THRESHOLD,
    # Or use pattern matching for groups
    DEFAULT_*,  # For default values
    BATTERY_*   # For battery-related constants
)
```

### Modifying Constants

When changing constants:

1. Update the value in `constants.py`
2. Add a comment with:
   - The source of the new value
   - Justification for the change
   - Date of the modification

### Adding New Constants

When adding new constants:

1. Place them in the appropriate section
2. Use the naming convention: `CATEGORY_PARAMETER_NAME`
3. Include a detailed comment explaining:
   - What the constant represents
   - Units of measurement
   - Source of the value
   - Uncertainty/variability if known

## Constants Documentation

The constants are organized into the following categories:

1. **Simulation parameters**: Grid size, time steps, etc.
2. **Agent population sizes**: Default numbers of different agent types
3. **Battery parameters**: Health thresholds, degradation rates, capacity
4. **Manufacturer parameters**: Production capacity, warranty policy
5. **Recycling facility parameters**: Processing capacity, efficiency
6. **Battery refurbisher parameters**: Technical capability, capacity
7. **EV Owner parameters**: Income, environmental consciousness, decision thresholds

## Adding Sources and References

When updating constants with real-world data, please add proper citations:

```python
# Battery degradation rate from Smith et al. (2023)
# https://doi.org/10.xxxx/xxxxx
DEFAULT_DEGRADATION_RATE = 0.0005  # 0.05% capacity loss per cycle
```
