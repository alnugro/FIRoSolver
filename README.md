# FIRoSolver

**FIRoSolver** is an FIR filter design tool with a focus on **multiplierless** implementation. It provides an interactive GUI to define and customize a filter’s magnitude response, then uses backend solvers (SAT via PySAT, SMT via Z3, or ILP via Gurobi) to find:

- **FIR coefficients** that meet your specified magnitude constraints.  
- An **optimal adder tree blueprint** (shown via a DOT file) to realize the filter multiplierlessly through bit shifts, adders, and optional DSP blocks in hardware.  
- **VHDL code** you can directly integrate into your FPGA design.

FIRoSolver generates partial or complete solutions for a given filter orders, wordlengths, DSP-block usage, and so on—both in “asserted” (manual parameter setting) mode and in fully “automatic” search mode (Gurobi only).

---

## Features

- **Interactive Magnitude Plotter**: Graphically define your desired filter’s magnitude response by dragging points on a plot.
- **Multiple Solvers Support, Though Mainly Gurobi**:  
  - **Gurobi** (ILP solver)  
  - **Z3** (SMT solver)  
  - **PySAT** (SAT solver)
- **Multiplierless Filter Design**:
  - Automatic generation of *adder trees* to implement constant multiplications with bit shifts and adders.
  - Optionally utilize DSP blocks for direct multiplications if your FPGA provides them.
- **Result Generation**:
  - DOT file for the adder tree blueprint.
  - VHDL code with user-defined wordlength and settings.
  - Coefficients that satisfy your magnitude response bounds.
- **Checkpointing**: Long running solves can be paused and resumed later.

---

## Installation

FIRoSolver is a Python application relying on PyQt for its GUI.

1. **Clone or Download** this repository.
2. **Install the required Python libraries** by running:
   
   ```bash
   pip install -r requirements.txt
   
  This should install (among others):
   - `numpy`
   - `matplotlib`
   - `python-sat`
   - `gurobipy`
   - `z3-solver`
   - `PyQt6`
   - `Pebble`
   - `filelock`
   - `scipy`
   - `psutil`

> **Note**: Gurobi requires a valid license. If you plan to use Gurobi as a solver, please ensure you have it properly installed and licensed.

---


## Getting Started

1. Run the GUI:
   
   ```bash
   python main.py
   
  This starts the full GUI version of FIRoSolver.

2. Run without GUI (experimental):
      python main_nogui.py
   This command-line approach is experimental and has limited functionality.

---

## How to Use

FIRoSolver’s GUI is organized into tabs at the bottom, typically used from **left to right**:

1. Magnitude Plotter
2. Asserted Search or Automatic Search
3. Optimization Setting
4. Solver Runs
5. Result Data

Below is a detailed guide for each tab.

---

### 1. Magnitude Plotter Tab

This is where you define the filter’s magnitude response using an **interactive GUI**.

- Frequency Range Table: Configure your frequency ranges (e.g., pass band, stop band and subsequently transition band).
- Interactive Graph: Drag the points on the plot to adjust the magnitude response bounds, right click to drag left click to move.
- Plot Wizard Subtab:
  - Offers undo/redo actions, selection tools, filtering options, and more.
  - Allows fine-tuned editing of the curve.
- Save/Load Plot:
  - Save your current magnitude response.
  - Load a previously saved curve.
  - The saved data only stores the *plot points* (frequency ranges cannot be re-edited after saving).

Once satisfied with the magnitude response, proceed to either **Asserted Search** or **Automatic Search**.

---

### 2. Asserted Search Tab

If you **already know** your filter’s parameters or want direct control, use **Asserted Search**:

- Filter Type & Order: Specify whether it’s filter type 1,2,3 or 4 FIR filter, and choose the filter order.
- Gain Bounds: Max and min gain allowed.
- Quick Check Satisfiable: (Gurobi-only) "Instantly" checks if your parameters are likely feasible before a full solve.
- DSP Block Availability:
  - Declare the number of DSP blocks you can use.
  - FIRoSolver will use these blocks for direct multiplication instead of generating an adder tree for those coefficients.
- Adder Depth: The maximum depth of the cascaded adder stages in your multiplierless design.
- Wordlength: Bit-length for your internal computations and coefficients.
- Adder Wordlength Extension: How many extra bits to allow inside the adder tree. A small number (0–2) is recommended.
- CM & Bounds Accuracy: Number of decimal places (precision) for floating-point calculations (higher = more accurate, but slower).
- Frequency Accuracy Multiplier: Controls the discretization of the frequency. More points → more accurate but slower.

Click "Quick Check Satisfiable" if using Gurobi to get a quick feasibility test. Otherwise, proceed to the next tab.

---

### 2 (Alternative). Automatic Search Tab

If you prefer FIRoSolver to **automatically** pick filter parameters (type/order) for you, use the **Automatic Search** tab.

- Automatic Search can attempt multiple filter orders and types to find an optimal solution under your magnitude constraints.
- You can choose a “real coefficient” simulation approach (no fixed wordlength in the solver), consequently:
  - Greatly increases runtime.
  - Potentially finds a more optimal adder tree.

Optimality is primarily judged by the total number of ripple-carry adders in the final design.

---

### 3. Optimization Setting Tab

Here you configure how the solvers run:

- Solver Option Subtab:
  - Set the number of threads for each solver (PySAT, Z3, Gurobi), anything >0 will be active solver
  - You can technically enable multiple solvers in parallel, but it’s **not recommended** because they might fight for resources.
- Search Options Subtab:
  - Error Prediction: Adds a conservative approach to error accumulation. Use only if encountering repeated “leaks” (where the design fails final constraints). Be warned it might yield “unsatisfiable” more often (not reccomended to use).
  - Deep Search: Pushes the solver for a more globally optimal solution. Often, a non-deep search is sufficient (recommended to turn on if you have beefy processor >30 threads).
  - Bound Transition Band: An experimental feature to add ceiling in the transition band. Increases the number of frequency points and thus solver runtime.
  - Worker Count & Search Step Size: For parallel search:
    - Worker Count: Number of solver workers in parallel (each worker itself should have ~6 threads if possible).
    - Search Step Size: The incremental step in adder count (or other parameters) to test feasibility. Larger steps can help skip infeasible ranges faster.

Finally, choose whether to use the **Asserted** or **Automatic** search approach with the switch at the bottom, then click the big red **Solve** button.

---

### 4. Solver Runs Tab

- Monitor Active/Queued Solves: Lists all solver tasks you’ve started, including those in progress and those completed.
- Pause and Resume: If a long run is in progress, you can stop it and resume later thanks to checkpointing. Even if FIRoSolver crashes, it can often resume from the last checkpoint.

---

### 5. Result Tab

After a solver run completes, the results appear here. There are two main subtabs:

1. Valid Result (No Leak):
   - The solution’s magnitude response **stays within** your specified bounds.
   - Here you’ll see:
     - A button to generate VHDL for the discovered filter.
     - A button to generate DOT data for the adder tree graph.
     - “Show Plot Data” to review the final frequency response vs. your target constraints.
2. Invalid Result (Leak):
   - Indicates the solver found an adder tree but the real frequency response (with discrete points) leaked outside the bounds.
   - Usually resolved by increasing frequency accuracy.
   - You can still generate VHDL, DOT, etc. to inspect the near-miss solution.

Result Settings Subtab:
- Wordlength options for VHDL: Final bit-width settings for the VHDL code generator.
  
Saving and Loading Results Subtab, There are 3 main files that will be generated per each run:
- problem_description.json: The input problem specification.
- result_valid.json and result_leak.json: The solver’s solutions.
- Use Save and Load to archive or restore your session’s results.

---

## Important Notes

- **Complexity**: This design problem is **NP-complete**, meaning solver runtimes increase **significantly** with higher filter order and tighter constraints:
  - **PySAT**: Orders above ~10 may take over an hour to solve.
  - **Z3**: Orders above ~24 may take over an hour to solve.
  - **Gurobi**: Orders above ~36 may take over an hour to solve.
  - Extremely large problems or tight constraints may require days or even weeks to find a solution. Please plan accordingly.
- **Checkpointing**: For long-running solves, the checkpoint/resume feature allows you to pause and continue solving at a later time, avoiding wasted computational effort.
- **Bugs & Feedback**: FIRoSolver is a complex tool, and while it has been tested, bugs may still exist. Please report any issues you find.
- **Future Optimizations**: Large filter designs are computationally demanding. I have ideas to optimize solver runtimes, such as separating the coefficient and adder tree problems. While this may not guarantee optimal solutions, it could significantly reduce runtime. These optimizations will only be implemented if there’s sufficient interest.

---

## Notice

FIRoSolver is part of my bachelor thesis. If you’d like to learn more about the underlying methods and implementation, you can refer to the following documents:

- **Thesis**: [FIRoSolver Thesis](https://drive.google.com/file/d/1pmxFTYDnh3lZfI-D73UZV6SxyQJdd2-Y/view?usp=drive_link)  
- **Errata**: [FIRoSolver Errata](https://drive.google.com/file/d/1YsRkj5yuX2vEf-r1EWtQrfi7AKv7Q--l/view?usp=sharing)

> **Note**: In the thesis, I stated that high-level formulations are slower than low-level formulations in Gurobi. Upon further testing, I found this is not universally true. In some cases, high-level formulations perform better. This could be an area for further improvement and exploration.

---








   
