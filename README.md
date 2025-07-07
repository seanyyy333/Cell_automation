# Cell_automation
Proof we live in a simulation down to the cell


import numpy as np
from scipy.stats import entropy
from ipywidgets import interact, IntSlider, fixed
import matplotlib.pyplot as plt
import random # Import random if not already imported

# Assuming run_cellular_automaton is defined elsewhere or in the preceding code
# If not, define a placeholder or the actual function here:
# def run_cellular_automaton(initial_state, generations, rule_number):
#     """Placeholder for the CA simulation function."""
#     grid_size = len(initial_state)
#     history = np.zeros((generations, grid_size), dtype=int)
#     history[0] = initial_state.copy()
#     # Implement the actual CA update logic based on rule_number
#     # This part needs to be provided or implemented based on elementary CA rules.
#     # Example placeholder (Rule 30 logic):
#     for t in range(1, generations):
#         for i in range(grid_size):
#             left = history[t-1, (i-1) % grid_size]
#             center = history[t-1, i]
#             right = history[t-1, (i+1) % grid_size]
#             # Example for Rule 30 (00111100 -> 0111000) in reverse neighbor order
#             # (right, center, left) -> new_state
#             if (right, center, left) == (1, 1, 1): new_state = 0
#             elif (right, center, left) == (1, 1, 0): new_state = 0
#             elif (right, center, left) == (1, 0, 1): new_state = 0
#             elif (right, center, left) == (1, 0, 0): new_state = 1
#             elif (right, center, left) == (0, 1, 1): new_state = 1
#             elif (right, center, left) == (0, 1, 0): new_state = 1
#             elif (right, center, left) == (0, 0, 1): new_state = 1
#             elif (right, center, left) == (0, 0, 0): new_state = 0
#             else: new_state = 0 # Should not happen with 0/1 states
#             # More general rule implementation using bitwise operations:
#             # state = (left << 2) | (center << 1) | right
#             # new_state = (rule_number >> state) & 1
#             history[t, i] = new_state # Use the correct new_state from actual rule logic
#
#     # Correct implementation using bitwise logic:
#     history = np.zeros((generations, grid_size), dtype=int)
#     history[0] = initial_state.copy()
#     for t in range(1, generations):
#         # Create padded state for easy neighborhood lookup with wrapping
#         padded_state = np.pad(history[t-1], (1, 1), mode='wrap')
#         for i in range(grid_size):
#             # Get neighborhood value (left, center, right) as a 3-bit integer
#             # left = padded_state[i]
#             # center = padded_state[i+1]
#             # right = padded_state[i+2]
#             # state = (left << 2) | (center << 1) | right # This matches Wolfram's convention 111=7, 110=6, ..., 000=0
#
#             # Or using direct indexing with wrapping:
#             left = history[t-1, (i - 1) % grid_size]
#             center = history[t-1, i]
#             right = history[t-1, (i + 1) % grid_size]
#             state = (left << 2) | (center << 1) | right # Neighborhood value from 0 to 7
#
#             # The new state is the state'th bit of the rule number
#             new_state = (rule_number >> state) & 1
#             history[t, i] = new_state
#
#     return history

# Function to explore with multiple random initial states - Modified to return history
def explore_random_initial_states(grid_size, generations, rule_number, num_simulations=3):
    """
    Runs and visualizes CA with multiple random initial states for a given rule.
    Returns the history of the last simulation.
    """
    print(f"\n--- Exploring Rule {rule_number} with {num_simulations} Random Initial States ---")
    last_history = None
    for sim_num in range(num_simulations):
        initial_state = np.random.randint(0, 2, size=grid_size)

        # Ensure at least one cell is initially active if you don't want all zeros
        # (Optional, depends on analysis goals)
        if not np.any(initial_state):
            initial_state[random.randint(0, grid_size - 1)] = 1

        print(f"\n--- Simulation {sim_num + 1} with Random Initial State ---")
        print("Initial State:", initial_state)

        # Run the simulation
        history = run_cellular_automaton(initial_state, generations, rule_number)
        last_history = history # Keep track of the last history

        # Visualize the history
        plt.figure(figsize=(10, generations / 5))
        plt.imshow(history, cmap='binary', interpolation='nearest')
        plt.title(f'Rule {rule_number}, Random Initial State {sim_num + 1}')
        plt.xlabel('Cell Index')
        plt.ylabel('Generation')
        plt.show()

    return last_history # Return the history of the last simulation

# Function to analyze rule behavior - Modified to use the history return value
def analyze_rule_behavior(grid_size, generations, rule_numbers_to_analyze):
    """Analyzes the behavior of different elementary cellular automaton rules."""
    for rule_number in rule_numbers_to_analyze:
        print(f"\n--- Analyzing Rule {rule_number} ---")

        # Use a simple initial state (e.g., a single '1' in the center) for consistent comparison
        initial_state = np.zeros(grid_size, dtype=int)
        initial_state[grid_size // 2] = 1
        print("Initial State:", initial_state)

        # Run the simulation
        history = run_cellular_automaton(initial_state, generations, rule_number)

        # Visualize the history
        plt.figure(figsize=(10, generations / 5))
        plt.imshow(history, cmap='binary', interpolation='nearest')
        plt.title(f'Behavior of Rule {rule_number}')
        plt.xlabel('Cell Index')
        plt.ylabel('Generation')
        plt.show()

        # Basic analysis
        active_cells_final = np.sum(history[-1])
        print(f"Number of active cells in the final generation: {active_cells_final}")

        # Return history so it can be used for metrics analysis outside the loop if needed
        # Or call analysis functions here directly
        analyze_complexity(history)
        analyze_advanced_metrics(history)


# Task 2: Complexity Analysis - Updated to include density plot
def analyze_complexity(history):
    """
    Analyzes the complexity of the cellular automaton history.
    Includes basic entropy calculation and density plot.
    """
    print("\n--- Basic Complexity Analysis ---")

    # Simple measure: Density of live cells over time
    density = np.sum(history, axis=1) / history.shape[1]
    plt.figure(figsize=(10, 4))
    plt.plot(density)
    plt.title('Density of Live Cells over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Density')
    plt.show()
    print("Density of live cells (first 10 generations):", density[:10])
    print("Density of live cells (last 10 generations):", density[-10:])


    # Entropy of the final generation (as a basic measure of randomness)
    final_generation = history[-1]
    counts = np.bincount(final_generation, minlength=2)
    probabilities = counts / np.sum(counts)
    probabilities = probabilities[probabilities > 0] # Filter out probabilities of 0

    if len(probabilities) > 1: # Need at least two outcomes to have entropy > 0
      shannon_entropy = entropy(probabilities, base=2)
      print(f"Shannon Entropy of the final generation: {shannon_entropy:.4f} bits")
    else:
        print("Cannot calculate entropy: Final generation is homogeneous (all 0s or all 1s).")


# Additional Metrics for Advanced Analysis (Recreating Universe)
def analyze_advanced_metrics(history):
    """
    Analyzes advanced metrics of the cellular automaton history,
    drawing concepts from complexity science, information theory, and physics.
    """
    print("\n--- Advanced Complexity and System Metrics Analysis ---")

    # Ensure history is a numpy array
    history = np.array(history)
    if history.ndim != 2:
        print("History must be a 2D array for advanced metrics.")
        return

    generations, grid_size = history.shape

    # --- Information Theory Metrics ---

    # Spatial Entropy (Entropy of rows/generations)
    spatial_entropies = []
    for generation in history:
        counts = np.bincount(generation, minlength=2)
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) > 1:
            spatial_entropies.append(entropy(probabilities, base=2))
        else:
            spatial_entropies.append(0)

    plt.figure(figsize=(10, 4))
    plt.plot(spatial_entropies)
    plt.title('Spatial Entropy over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Entropy (bits)')
    plt.show()
    print(f"Average Spatial Entropy: {np.mean(spatial_entropies):.4f}")
    print(f"Final Spatial Entropy: {spatial_entropies[-1]:.4f}")


    # Temporal Entropy (Entropy of columns/cell histories)
    temporal_entropies = []
    history_transposed = history.T
    for cell_history in history_transposed:
        counts = np.bincount(cell_history, minlength=2)
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) > 1:
            temporal_entropies.append(entropy(probabilities, base=2))
        else:
            temporal_entropies.append(0)

    if temporal_entropies: # Check if list is not empty
        plt.figure(figsize=(10, 4))
        plt.hist(temporal_entropies, bins=20)
        plt.title('Distribution of Temporal Entropy Across Cells')
        plt.xlabel('Entropy (bits)')
        plt.ylabel('Frequency')
        plt.show()
        print(f"Average Temporal Entropy: {np.mean(temporal_entropies):.4f}")
    else:
        print("Could not calculate temporal entropy.")


    # Mutual Information (Between adjacent cells in the final generation)
    if grid_size > 1:
        final_generation = history[-1]
        # Calculate joint probability distribution for adjacent pairs (00, 01, 10, 11)
        joint_counts = np.zeros((2, 2))
        # Using toroidal wrapping for the last pair
        for i in range(grid_size):
            j = (i + 1) % grid_size
            joint_counts[final_generation[i], final_generation[j]] += 1

        total_pairs = np.sum(joint_counts)
        if total_pairs > 0:
            joint_probabilities = joint_counts / total_pairs

            # Calculate individual probabilities
            p_0 = np.sum(final_generation == 0) / grid_size
            p_1 = 1 - p_0

            mutual_info = 0
            for i in range(2):
                for j in range(2):
                    p_xy = joint_probabilities[i, j]
                    if p_xy > 0: # Avoid log(0)
                        p_x = (np.sum(final_generation == i)) / grid_size # Marginal P(X=i)
                        p_y = (np.sum(final_generation == j)) / grid_size # Marginal P(Y=j)

                        # P(Y=j) is the same as P(X=j) in the next position assuming stationarity
                        # A more rigorous approach would sum over all pairs.

                        # Using the marginal probabilities derived from the final generation
                        p_x_marginal = np.sum(joint_counts[i, :]) / total_pairs
                        p_y_marginal = np.sum(joint_counts[:, j]) / total_pairs


                        if p_x_marginal > 0 and p_y_marginal > 0:
                             mutual_info += p_xy * np.log2(p_xy / (p_x_marginal * p_y_marginal))
                        # else: if marginal is 0, joint must be 0, handled by p_xy > 0

            print(f"Pairwise Mutual Information (Adjacent Cells, Final Generation): {mutual_info:.4f}")
        else:
            print("Cannot calculate pairwise mutual information: No pairs found.")
    else:
        print("Cannot calculate pairwise mutual information: Grid size is too small.")


    # --- Physics-inspired Metrics ---

    # Energy (Simple analog: number of 'active' cells)
    energy_over_time = np.sum(history, axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(energy_over_time)
    plt.title('Total Active Cells (Simple Energy Analog) over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Number of Active Cells')
    plt.show()

    # Fluctuation/Variance
    if generations > 1:
        print(f"Variance in Active Cells over Generations: {np.var(energy_over_time):.4f}")
    else:
        print("Cannot calculate variance: Not enough generations.")

    # Correlation Length (Spatial Correlation)
    if grid_size > 1:
        final_generation = history[-1]
        mean_state = np.mean(final_generation)
        # Calculate correlation function C(d) = <s_i * s_{i+d}> - <s_i>^2
        # Averaged over the final generation with toroidal wrapping.
        max_distance = grid_size // 2 # Consider distances up to half the grid size
        correlations = []
        distances = []

        for d in range(1, max_distance + 1): # Include distance 1 up to max_distance
             # Calculate average product for cells distance 'd' apart
             product_sum = 0
             for i in range(grid_size):
                 j = (i + d) % grid_size
                 product_sum += final_generation[i] * final_generation[j]
             avg_product = product_sum / grid_size # Average over all starting points i

             correlation = avg_product - mean_state**2
             correlations.append(correlation)
             distances.append(d)

        if distances:
            plt.figure(figsize=(10, 4))
            plt.plot(distances, correlations, marker='o', linestyle='-')
            plt.title('Spatial Correlation Function (Final Generation)')
            plt.xlabel('Distance')
            plt.ylabel('Correlation')
            plt.grid(True)
            plt.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add line at 0 correlation
            plt.show()
            # Estimating correlation length (e.g., where correlation drops to 1/e or 0)
            # is more complex and depends on the specific decay function.
        else:
            print("Cannot calculate spatial correlation: Grid size too small.")
    else:
        print("Cannot calculate spatial correlation: Grid size too small.")

    # --- Periodicity Detection (Basic Check) ---
    # Can the final generation be tiled to form the history? Or does the grid state repeat?
    # Checking for repeating states in the *entire* grid history is a basic approach.
    # More advanced methods involve finding the shortest repeating sequence.
    print("\n--- Periodicity Check ---")
    if generations > 1:
        # Convert each generation (row) to a hash or tuple for comparison
        # Using tuple conversion which is hashable
        history_tuples = [tuple(row) for row in history]

        # Check for repeated states from the end of the history
        repeated_state_found = False
        for i in range(generations - 2, -1, -1): # Iterate backwards from the second to last generation
            if history_tuples[i] == history_tuples[-1]:
                period = generations - 1 - i
                print(f"Detected potential periodicity: State in generation {i} is the same as final state (generation {generations - 1}). Period: {period} generations.")
                repeated_state_found = True
                break # Found the most recent repeat

        if not repeated_state_found:
            print("No simple repetition of the final state detected in history.")
    else:
        print("Cannot check for periodicity: Not enough generations.")


# Task 4: Higher Dimensions (2D Cellular Automata - Conway's Game of Life) - Add Metrics
# Analyzing 2D CA requires adapting metrics. Spatial entropy can be 2D.
# Temporal entropy applies to each cell's history. Correlation length is 2D.

def analyze_game_of_life_metrics(history_2d):
    """
    Analyzes metrics for Game of Life history.
    Assumes history_2d is a list of 2D numpy arrays (states).
    """
    print("\n--- Game of Life Metrics Analysis ---")

    if not history_2d:
        print("No history to analyze.")
        return

    generations = len(history_2d)
    rows, cols = history_2d[0].shape

    # Density over time
    densities = [np.mean(grid) for grid in history_2d]
    plt.figure(figsize=(10, 4))
    plt.plot(densities)
    plt.title('Density of Live Cells over Generations (Game of Life)')
    plt.xlabel('Generation')
    plt.ylabel('Density')
    plt.show()
    print(f"Average Density: {np.mean(densities):.4f}")
    print(f"Final Density: {densities[-1]:.4f}")


    # Spatial Entropy (2D) - Using flattened grid as a simple measure
    spatial_entropies_2d_flat = []
    for grid in history_2d:
        flat_grid = grid.flatten()
        counts = np.bincount(flat_grid, minlength=2)
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) > 1:
            spatial_entropies_2d_flat.append(entropy(probabilities, base=2))
        else:
            spatial_entropies_2d_flat.append(0)

    plt.figure(figsize=(10, 4))
    plt.plot(spatial_entropies_2d_flat)
    plt.title('Spatial Entropy (Flattened Grid) over Generations (Game of Life)')
    plt.xlabel('Generation')
    plt.ylabel('Entropy (bits)')
    plt.show()
    print(f"Average Spatial Entropy (Flattened): {np.mean(spatial_entropies_2d_flat):.4f}")
    print(f"Final Spatial Entropy (Flattened): {spatial_entropies_2d_flat[-1]:.4f}")


    # Temporal Entropy per cell
    if generations > 1:
        temporal_entropies_2d = []
        for i in range(rows):
            for j in range(cols):
                cell_history = np.array([grid[i, j] for grid in history_2d])
                counts = np.bincount(cell_history, minlength=2)
                probabilities = counts / np.sum(counts)
                probabilities = probabilities[probabilities > 0]
                if len(probabilities) > 1:
                    temporal_entropies_2d.append(entropy(probabilities, base=2))
                else:
                    temporal_entropies_2d.append(0)

        if temporal_entropies_2d:
            plt.figure(figsize=(10, 4))
            plt.hist(temporal_entropies_2d, bins=20)
            plt.title('Distribution of Temporal Entropy Across Cells (Game of Life)')
            plt.xlabel('Entropy (bits)')
            plt.ylabel('Frequency')
            plt.show()
            print(f"Average Temporal Entropy (per cell): {np.mean(temporal_entropies_2d):.4f}")
        else:
             print("Could not calculate temporal entropy.")
    else:
        print("Cannot calculate temporal entropy: Not enough generations.")


    # Correlation Length (2D) - More complex, involves distance in 2D grid
    # Calculation of 2D correlation function C(dx, dy) is non-trivial here.
    # For a basic measure, one could average correlations over radial distance.
    # Skipping full 2D correlation function for brevity.

    # Periodicity - Detect if the grid state enters a cycle
    print("\n--- Periodicity Check (Game of Life) ---")
    if generations > 1:
        # Convert each grid (state) to a hash or flatten and convert to tuple
        history_hashes = [grid.tobytes() for grid in history_2d] # Using tobytes for hashing

        # Check for repeated states from the end
        repeated_state_found = False
        for i in range(generations - 2, -1, -1):
            if history_hashes[i] == history_hashes[-1]:
                period = generations - 1 - i
                print(f"Detected potential periodicity: State in generation {i} is the same as final state (generation {generations - 1}). Period: {period} generations.")
                repeated_state_found = True
                break

        if not repeated_state_found:
            print("No simple repetition of the final state detected in history.")
    else:
        print("Cannot check for periodicity: Not enough generations.")


# Modify run_game_of_life to store and return history
def run_game_of_life_with_history(grid_size_2d=(50, 50), generations_2d=100, initial_density=0.1):
    """Runs, visualizes, and returns history of Conway's Game of Life."""
    print("\n--- Running Conway's Game of Life (2D CA) ---")

    initial_grid = np.random.choice([0, 1], size=grid_size_2d, p=[1 - initial_density, initial_density])
    current_grid = initial_grid.copy()

    history_2d = [initial_grid.copy()] # Store initial state

    plt.figure(figsize=(8, 8))
    plt.imshow(current_grid, cmap='binary', interpolation='nearest')
    plt.title('Game of Life - Initial State')
    plt.show()

    for i in range(generations_2d):
        current_grid = update_game_of_life(current_grid)
        history_2d.append(current_grid.copy())

    plt.figure(figsize=(8, 8))
    plt.imshow(current_grid, cmap='binary', interpolation='nearest')
    plt.title(f'Game of Life - Final State (Generation {generations_2d})')
    plt.show()

    return history_2d

# Task 5: Real-World Applications (Example: Basic Diffusion Model) - Add Metrics
# Metrics for diffusion might include:
# - Mean position of the substance (if applicable)
# - Spread/Variance of the substance distribution
# - Rate of diffusion (how quickly it spreads)
# - Total amount of substance (should be conserved in this simple model)

def analyze_diffusion_metrics(history_diffusion):
    """Analyzes metrics for Diffusion simulation history."""
    print("\n--- Diffusion Simulation Metrics Analysis ---")

    if not history_diffusion:
        print("No history to analyze.")
        return

    history_diffusion = np.array(history_diffusion)
    generations, grid_size = history_diffusion.shape


    # Total amount of substance over time
    total_substance = np.sum(history_diffusion, axis=1)
    plt.figure(figsize=(10, 4))
    plt.plot(total_substance)
    plt.title('Total Substance over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Total Substance')
    plt.show()
    print(f"Initial Total Substance: {total_substance[0]:.4f}")
    print(f"Final Total Substance: {total_substance[-1]:.4f}")
    print(f"Change in Total Substance: {total_substance[-1] - total_substance[0]:.4f}") # Should be close to 0 for conserved models

    # Variance of the distribution over time (measure of spread)
    # Calculate weighted variance: Sum( (i - mean)^2 * concentration_i ) / Sum(concentration_i)
    variances = []
    generations_list = []
    for i in range(generations):
        current_state = history_diffusion[i]
        total = np.sum(current_state)
        if total > 1e-9: # Use a small epsilon to avoid division by near zero
            # Calculate mean position (using indices as positions)
            mean_pos = np.sum(current_state * np.arange(grid_size)) / total
            # Calculate variance
            variance = np.sum(current_state * (np.arange(grid_size) - mean_pos)**2) / total
            variances.append(variance)
            generations_list.append(i)
        else:
            # If total substance is zero or near zero, variance is also zero.
            variances.append(0)
            generations_list.append(i)


    if variances:
        plt.figure(figsize=(10, 4))
        plt.plot(generations_list, variances)
        plt.title('Spatial Variance of Substance Distribution over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Variance')
        plt.show()
        print(f"Initial Variance: {variances[0]:.4f}")
        print(f"Final Variance: {variances[-1]:.4f}")
        # A simple estimate of the diffusion constant could be related to the slope
        # of the variance vs. time plot (Variance ~ 2 * D * t in simple cases).
        if len(generations_list) > 1:
            # Simple linear fit to estimate rate (very rough)
            slope, intercept = np.polyfit(generations_list, variances, 1)
            print(f"Estimated Rate of Variance Increase (approx. 2*Diffusion Constant): {slope:.4f}")


    # Spatial Entropy of the concentration profile
    # Treat concentrations as probabilities (normalize by total substance)
    spatial_entropies_conc = []
    for i in range(generations):
        current_state = history_diffusion[i]
        total = np.sum(current_state)
        if total > 1e-9:
            probabilities = current_state / total
            probabilities = probabilities[probabilities > 0]
            if len(probabilities) > 1:
                spatial_entropies_conc.append(entropy(probabilities, base=2))
            else:
                spatial_entropies_conc.append(0) # If only one cell has substance
        else:
            spatial_entropies_conc.append(0) # If no substance, entropy is 0

    if spatial_entropies_conc:
        plt.figure(figsize=(10, 4))
        plt.plot(generations_list, spatial_entropies_conc)
        plt.title('Spatial Entropy of Concentration Profile over Generations')
        plt.xlabel('Generation')
        plt.ylabel('Entropy (bits)')
        plt.show()
        print(f"Initial Spatial Entropy (Concentration): {spatial_entropies_conc[0]:.4f}")
        print(f"Final Spatial Entropy (Concentration): {spatial_entropies_conc[-1]:.4f}")
        # Entropy increases as the substance spreads and becomes more uniform.


# Modify run_diffusion_1d to store and return history
def run_diffusion_1d_with_history(grid_size, generations, diffusion_rate=0.2):
    """Simulates 1D diffusion, visualizes, and returns history."""
    print("\n--- Running 1D Diffusion Simulation ---")

    initial_state = np.zeros(grid_size)
    # Ensure the initial state is set correctly
    center_start = max(0, grid_size // 2 - 5)
    center_end = min(grid_size, grid_size // 2 + 5)
    initial_state[center_start : center_end] = 100 # Initial concentration

    current_state = initial_state.copy()
    history_diffusion = [current_state.copy()]

    # Initial state plot
    plt.figure(figsize=(10, 3)) # Adjust figure size for 1D
    plt.plot(initial_state)
    plt.title('1D Diffusion Simulation - Initial State')
    plt.xlabel('Cell Index')
    plt.ylabel('Concentration')
    plt.ylim(0, np.max(initial_state) * 1.1) # Set consistent y-limit
    plt.grid(True)
    plt.show()

    # History visualization as an image
    # Make history a 2D array with generations as rows, cells as columns
    history_image_data = [current_state.copy()]

    for i in range(generations):
        next_state = current_state.copy()
        for j in range(grid_size):
            left_neighbor = (j - 1 + grid_size) % grid_size
            right_neighbor = (j + 1) % grid_size

            # Fick's first law analog: flow is proportional to concentration gradient
            # Change in concentration at j is proportional to the difference
            # between its neighbors and itself.
            change = diffusion_rate * (
                (current_state[left_neighbor] - current_state[j]) +
                (current_state[right_neighbor] - current_state[j])
            )
            next_state[j] += change
            next_state[j] = max(0, next_state[j]) # Prevent negative concentrations

        current_state = next_state
        history_image_data.append(current_state.copy()) # Store state for image

    # Visualize the diffusion process over time using imshow
    plt.figure(figsize=(10, generations / 20)) # Adjust figure size based on generations
    plt.imshow(np.array(history_image_data), cmap='viridis', interpolation='nearest', aspect='auto')
    plt.title(f'1D Diffusion Simulation over Generations (Rate={diffusion_rate})')
    plt.xlabel('Cell Index')
    plt.ylabel('Generation')
    plt.colorbar(label='Concentration')
    plt.gca().invert_yaxis() # Invert y-axis so generation 0 is at the top
    plt.show()

    # Visualize the final state as a plot
    plt.figure(figsize=(10, 3)) # Adjust figure size for 1D
    plt.plot(current_state)
    plt.title(f'1D Diffusion Simulation - Final State (Generation {generations})')
    plt.xlabel('Cell Index')
    plt.ylabel('Concentration')
    plt.ylim(0, np.max(initial_state) * 1.1) # Use initial state max for consistent scale
    plt.grid(True)
    plt.show()


    return history_image_data # Return the history


# Ensure run_cellular_automaton is defined before calling it
# Define the run_cellular_automaton function here if it was not in the preceding code:
def run_cellular_automaton(initial_state, generations, rule_number):
    """Runs an elementary cellular automaton simulation."""
    grid_size = len(initial_state)
    history = np.zeros((generations, grid_size), dtype=int)
    history[0] = initial_state.copy()

    for t in range(1, generations):
        # Using direct indexing with wrapping for neighborhood lookup
        current_state = history[t-1]
        for i in range(grid_size):
            left = current_state[(i - 1) % grid_size]
            center = current_state[i]
            right = current_state[(i + 1) % grid_size]

            # Calculate the neighborhood value (0-7)
            state_value = (left << 2) | (center << 1) | right

            # Get the new state based on the rule number's bit corresponding to the state value
            new_state = (rule_number >> state_value) & 1
            history[t, i] = new_state

    return history


# --- Example Usage - Integrated with Advanced Metrics ---

# Define parameters
grid_size = 150  # Increased grid size
generations = 200 # Increased generations
rule_to_explore = 110 # A common complex rule
rules_to_compare = [30, 110, 184, 90, 54, 60] # Examples of different rule types

print("--- Starting Simulations and Analysis ---")

# 1. Explore with a single initial state (e.g., single '1')
initial_state_single_cell = np.zeros(grid_size, dtype=int)
initial_state_single_cell[grid_size // 2] = 1
print(f"\n--- Running Simulation with Single Cell Initial State (Rule {rule_to_explore}) ---")
history_single_cell = run_cellular_automaton(initial_state_single_cell, generations, rule_to_explore)

plt.figure(figsize=(10, generations / 5))
plt.imshow(history_single_cell, cmap='binary', interpolation='nearest')
plt.title(f'Rule {rule_to_explore}, Single Cell Initial State')
plt.xlabel('Cell Index')
plt.ylabel('Generation')
plt.show()

analyze_complexity(history_single_cell) # Basic complexity
analyze_advanced_metrics(history_single_cell) # Advanced metrics

# 2. Explore with multiple random initial states
print("\n--- Exploring with Random Initial States ---")
# We'll run and analyze the last simulation's history for demonstration
# The explore_random_initial_states function now returns the history of the last run
last_random_history = explore_random_initial_states(grid_size, generations, rule_to_explore, num_simulations=3)
if last_random_history is not None:
    print("\n--- Analysis for the Last Random Initial State Simulation ---")
    analyze_complexity(last_random_history) # Basic complexity for the last random sim
    analyze_advanced_metrics(last_random_history) # Advanced metrics for the last random sim


# 3. Analyze the behavior of different rules
print("\n--- Analyzing Behavior of Different Rules ---")
# The analyze_rule_behavior function now includes metric calls inside its loop
analyze_rule_behavior(grid_size, generations, rules_to_compare)


# Task 3: Interactive Exploration (using ipywidgets) - Updated
# Ensure ipywidgets is installed and enabled for your environment
# !pip install ipywidgets
# !jupyter nbextension enable --py widgetsnbextension # Run this if in classic Jupyter
# !jupyter labextension enable @jupyter-widgets/jupyterlab-manager # Run this if in JupyterLab


# Update the interactive_ca function to include advanced metrics analysis
# This function is designed for interactive use, so it runs the simulation and analysis when parameters change.
def interactive_ca_with_metrics(rule_number, grid_size, generations):
    """Interactive function to run, display CA, and analyze with metrics."""
    print(f"\n--- Running Interactive Simulation: Rule {rule_number}, Grid Size {grid_size}, Generations {generations} ---")
    initial_state = np.zeros(grid_size, dtype=int)
    initial_state[grid_size // 2] = 1  # Start with a single cell

    history = run_cellular_automaton(initial_state, generations, rule_number)

    plt.figure(figsize=(10, generations / 5))
    plt.imshow(history, cmap='binary', interpolation='nearest')
    plt.title(f'Interactive CA - Rule {rule_number}')
    plt.xlabel('Cell Index')
    plt.ylabel('Generation')
    plt.show()

    analyze_complexity(history)
    analyze_advanced_metrics(history)


print("\n--- Interactive Exploration (with Metrics) ---")
# Display the interactive controls
interact(interactive_ca_with_metrics,
         rule_number=IntSlider(min=0, max=255, step=1, value=110, description='Rule Number:'),
         grid_size=IntSlider(min=50, max=500, step=10, value=150, description='Grid Size:'),
         generations=IntSlider(min=10, max=250, step=10, value=200, description='Generations:'));


# Task 4: Higher Dimensions (2D Cellular Automata - Conway's Game of Life) - Add Metrics

print("\n--- Running Conway's Game of Life (2D CA) and Analysis ---")
game_of_life_history = run_game_of_life_with_history(grid_size_2d=(100, 100), generations_2d=250, initial_density=0.15)
analyze_game_of_life_metrics(game_of_life_history)


# Task 5: Real-World Applications (Example: Basic Diffusion Model) - Add Metrics

print("\n--- Running 1D Diffusion Simulation and Analysis ---")
diffusion_history = run_diffusion_1d_with_history(grid_size=300, generations=300, diffusion_rate=0.1)
analyze_diffusion_metrics(diffusion_history)

print("\n--- Simulation and Analysis Complete ---")
