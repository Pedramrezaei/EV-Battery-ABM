# EV Battery Circularity ABM

Agent-based model to study electric vehicle battery circularity in the Netherlands, focusing on second-life applications and the goal of 100% circularity by 2050.

## Project Structure

- `src/`: Source code for the ABM
  - `agents/`: Agent class definitions
  - `models/`: Core model implementation
  - `utils/`: Utility functions and helpers
  - `visualization/`: Visualization components
- `data/`: Input data and parameters
- `docs/`: Documentation and model description
- `tests/`: Unit tests
- `notebooks/`: Jupyter notebooks for analysis and visualization

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Model

There are two ways to run the simulation:

### 1. Interactive Matplotlib Visualization

Run the simulation with a real-time matplotlib visualization:

```bash
cd ev_battery_abm
python src/run_simulation.py
```

This will:

- Run the simulation for 50 steps
- Show an animated visualization with:
  - Agent locations on the grid
  - Battery status over time
  - Circularity rates
  - Materials recovered
  - Battery age distribution

### 2. Mesa Web Interface (Advanced)

For more interactive control, you can use the Mesa web interface:

```bash
cd ev_battery_abm
python src/run.py
```

This will:

- Start a web server on port 8521
- Open a browser with the Mesa visualization
- Allow parameter adjustment with sliders
- Provide step-by-step control

## Key Components

1. Agent Types:

   - EV Owners: Consumers who use and make decisions about batteries
   - Car Manufacturers: Produce batteries and handle take-backs
   - Recycling Facilities: Process end-of-life batteries for material recovery
   - Battery Refurbishers: Convert suitable batteries for second-life applications

2. Key Processes:
   - Primary battery lifecycle (production, use, degradation)
   - End-of-first-life decisions (recycle, refurbish, keep)
   - Second-life applications (grid storage, etc.)
   - Final recycling and material recovery

## License

[To be determined]
