using Distributed
@everywhere using Random, PyCall
@everywhere @pyimport matplotlib.pyplot as plt
@everywhere @pyimport numpy as np

# Parameters
@everywhere const Nx = 50
@everywhere const Ny = 50
@everywhere const eqSteps = 40000
@everywhere T = [0.2]
@everywhere nt = length(T)
@everywhere J = 1.0

# Initialize spin lattice with random angles between 0 and 2π
@everywhere function initial_state(Nx, Ny; seed=100)
    Random.seed!(seed)
    return 2 * pi * rand(Nx, Ny)
end

# Energy calculation: interaction between nearest neighbors
@everywhere function calc_energy(lattice)
    energy = 0.0
    for i in 1:Nx
        for j in 1:Ny
            θ = lattice[i, j]
            # Nearest neighbors with periodic boundary conditions
            θ_right = lattice[mod1(i+1, Nx), j]
            θ_left  = lattice[mod1(i-1, Nx), j]
            θ_up    = lattice[i, mod1(j+1, Ny)]
            θ_down  = lattice[i, mod1(j-1, Ny)]
            # Interaction energy
            energy -= J * (cos(θ - θ_right) + cos(θ - θ_left) +
                            cos(θ - θ_up) + cos(θ - θ_down))
        end
    end
    return energy * 0.5  # To avoid double counting of pairs
end

# Magnetization calculation: sum of the spins
@everywhere function calc_magnetization_x(lattice)
    Mx = cos.(lattice)  # X component of the magnetization
    M = sum(Mx) # Normalized magnetization
    return M / eqSteps
end

@everywhere function calc_magnetization_y(lattice)
    My = sin.(lattice)  # X component of the magnetization
    M = sum(My) # Normalized magnetization
    return M / eqSteps
end

# Monte Carlo move using Metropolis algorithm
@everywhere function mcmove(lattice, beta)
    for _ in 1:(Nx * Ny)
        i = rand(1:Nx)
        j = rand(1:Ny)
        θ_old = lattice[i, j]
        θ_new = θ_old + (rand() - 0.5) * 0.05 # Small random angle change

        E_old = -J * (cos(θ_old - lattice[mod1(i+1, Nx), j]) + cos(θ_old - lattice[mod1(i-1, Nx), j]) +
                    cos(θ_old - lattice[i, mod1(j+1, Ny)]) + cos(θ_old - lattice[i, mod1(j-1, Ny)]))
        E_new = -J * (cos(θ_new - lattice[mod1(i+1, Nx), j]) + cos(θ_new - lattice[mod1(i-1, Nx), j]) +
                    cos(θ_new - lattice[i, mod1(j+1, Ny)]) + cos(θ_new - lattice[i, mod1(j-1, Ny)]))
        
        dE = E_new - E_old

        # Metropolis criterion
        if (dE < 0) 
            lattice[i, j] = θ_new
        elseif rand() < exp(-dE * beta)
            lattice[i, j] = θ_new  # Accept move
        end
    end
    return lattice
end

function calc_correlation_function(lattice)
    correlation = 0.0  # Initialize correlation value for r = 1
    
    for i in 1:Nx
        for j in 1:Ny
            θ_i = lattice[i, j]  # Angle at site (i, j)
            
            # Find neighbors at distance r = 1 using periodic boundary conditions
            iu = mod1(i+1, Nx)  
            ju = mod1(j+1, Ny)  
            id = mod1(i-1, Nx)
            jd = mod1(j-1, Ny)

            # Spin angles at neighbors
            θ_right = lattice[iu, j]
            θ_left = lattice[id, j]
            θ_up = lattice[i, ju]
            θ_down = lattice[i, jd]
            
            # Dot products between spin vectors at distance r = 1
            correlation += abs(cos(θ_i - θ_right)) + abs(cos(θ_i - θ_left)) + abs(cos(θ_i - θ_up)) + abs(cos(θ_i - θ_down))
        end
    end

    # Total number of pairs considered
    total_pairs = 4 * Nx * Ny
    
    # Normalize by the total number of pairs
    correlations = correlation / total_pairs

    return correlations  # Return the correlation for r = 1
end

function main()
    # Initialize vectors to store Energy, and Magnetization values and corresponding sweep counts
    E_vals = Float64[]
    M_vals_x = Float64[]
    M_vals_y = Float64[]
    correlation_vals = Float64[]
    sweep_counts = Int[]  # Sweep counts for all calculations (Energy, Magnetization)
    
    for ti in 1:nt
        T_val = T[ti]
        beta = 1.0 / T_val

        # Initialize lattice
        lattice = initial_state(Nx, Ny; seed=50)
        nx_p = cos.(lattice)  # X-component of spins
        ny_p = sin.(lattice)  # Y-component of spins

        # Equilibration phase: Calculate Energy, and Magnetization for every sweep
        for i in 1:eqSteps
            lattice = mcmove(lattice, beta)

            # Update spin components
            nx_p = cos.(lattice)
            ny_p = sin.(lattice)
            
            # Calculate energy and magnetization
            energy = calc_energy(lattice)
            magnetization_x = calc_magnetization_x(lattice)
            magnetization_y = calc_magnetization_y(lattice)

            # Calculate the correlation function
            correlations = calc_correlation_function(lattice)

            # Store Energy, Magnetization values and corresponding sweep count
            push!(E_vals, energy)
            push!(M_vals_x, magnetization_x)
            push!(M_vals_y, magnetization_y)
            push!(correlation_vals, correlations)
            push!(sweep_counts, i)  # Store sweep count for all calculations
        end
        
        # After the final sweep, visualize the lattice and save the image
        filename = "XY_T$(T_val)_final.png"
        visualize_lattice(lattice, filename)
    end
    return sweep_counts, E_vals, M_vals_x, M_vals_y, correlation_vals
end

# Visualization of final configuration
function visualize_lattice(lattice, filename)
    x, y = np.meshgrid(1:Nx, 1:Ny)
    U = cos.(lattice)  # X-component of spins
    V = sin.(lattice)  # Y-component of spins
    
    plt.figure(figsize=(8, 8))
    plt.quiver(x, y, U, V, pivot = "mid")  # Plot spins as arrows, adjust scale as needed
    plt.title("2D XY Model Spin Configuration")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    #plt.savefig(filename)
    plt.show()
end

# Function to plot E, M, correlation vs sweeps
function plot_results(sweep_counts, E_vals, M_vals_x, M_vals_y, correlation_vals)

    plt.figure(figsize=(8, 6))
    plt.scatter(sweep_counts, E_vals, marker="o", label="Energy vs Sweeps", color="red", s=1)
    plt.title("Energy vs Number of Sweeps")
    plt.xlabel("Number of Sweeps")
    plt.ylabel("Energy")
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.scatter(sweep_counts, M_vals_x, marker="o", label="Magnetization X vs Sweeps", color="blue", s=1)
    plt.scatter(sweep_counts, M_vals_y, marker="o", label="Magnetization Y vs Sweeps", color="orange", s=1)
    plt.title("Magnetization (X and Y) vs Number of Sweeps")
    plt.xlabel("Number of Sweeps")
    plt.ylabel("Magnetization")
    plt.legend()

    plt.figure(figsize=(8, 6))
    plt.scatter(sweep_counts, correlation_vals, marker="o", label="Correlation at r=1", color="green", s=1)
    plt.title("Correlation Function at Distance r=1 vs Number of Sweeps")
    plt.xlabel("Number of Sweeps")
    plt.ylabel("Correlation Function C(r=1)")
    plt.show()
end


sweep_counts, E_vals, M_vals_x, M_vals_y, correlation_vals = main()
plot_results(sweep_counts, E_vals, M_vals_x, M_vals_y, correlation_vals)



