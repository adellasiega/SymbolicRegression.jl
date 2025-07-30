import Pkg
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Distances")
Pkg.add("MLJBase")
using SymbolicRegression
using SymbolicRegression: ValidVector
using StatsBase
using Plots
using Distances
import Base: min
using MLJBase: machine, fit!, predict, report

"""
    simulate(drift_fn::Function, y0::Vector{Float64}, dt=0.1, t_end=10.0) -> Matrix{Float64}

Simulates stochastic differential equations (SDEs) of the form:
    dy_t = drift_fn(y)dt + 0.1 * dW_t

Arguments:
    drift_fn: a scalar function R -> R (applied element-wise to the vector y)
    y0: a vector in R^N, containing the N initial conditions
    dt: integration time step (default 0.1)
    t_end: final time of integration (default 10.0)

Returns:
    trajectories: a (T, N) matrix where each column represents a trajectory and T = t_end / dt
"""
function simulate(drift_fn::Function, diff_fn::Function, y0::Vector{Float64}, dt=0.1, t_end=10.0):: Matrix{Float64}
    N = length(y0)
    T = Int(t_end/dt)
    trajectories = Array{Float64, 2}(undef, T, N)
    trajectories[1, :] .= y0
    y = copy(y0)
    sqrtdt = sqrt(dt)
    for t in 2:T
        y .= y .+ drift_fn(y) .* dt .+ diff_fn(y) .* randn(N) .* sqrtdt
        y .= clamp.(y, -1e3, 1e3)
        trajectories[t, :] = y
    end
    return trajectories
end

"""
    flatten_matrix(M::Matrix{Float64}) -> Matrix{Float64}

Flattens a (T, N) matrix of trajectories into a (1, T*N) row vector matrix.

Arguments:
    M: a matrix of shape (T, N) representing N trajectories over T time steps

Returns:
    A matrix of shape (1, T*N) containing the flattened data
"""
function flatten_matrix(M::Matrix{Float64})::Matrix{Float64}
    return reshape(vec(M)', 1, :) # (1, T*N)
end

"""
    unflatten_matrix(flat_M::Matrix{Float64}, N::Int) -> Matrix{Float64}

Unflattens a (1, T*N) matrix back to shape (T, N).

Arguments:
    flat_M: a matrix of shape (1, T*N), containing flattened trajectory data
    N: the number of variables (trajectories) originally in the data

Returns:
    A matrix of shape (T, N) representing N trajectories over T time steps
"""
function unflatten_matrix(flat_M::Matrix{Float64}, N::Int)::Matrix{Float64}
    T = size(flat_M,2) ÷ N
    return reshape(flat_M, T, N)  # (T, N)
end

"""
    compute_distance(X::Matrix{Float64}, Y::Matrix{Float64}) -> Float64

Computes the average L1 distance between the empirical cumulative distribution functions (ECDFs) of
corresponding time slices (rows) from two matrices.

Arguments:
    X: a (T, N) matrix of simulated or true trajectories
    Y: a (T, N) matrix of simulated or true trajectories to compare

Returns:
    A scalar representing the average ECDF L1 distance over all time steps.
    Returns a large penalty (1e10) if NaNs or Infs are present in the data.
"""
function compute_distance(X::Matrix{Float64}, Y::Matrix{Float64})::Float64
    T = size(X, 1)
    tot = 0.0
    for t in 1:T
        x = X[t, :]
        y = Y[t, :]
        if any(isnan.(x)) || any(isnan.(y)) || any(isinf.(x)) || any(isinf.(y))
	        return 1e10
        end
        ecdf1 = ecdf(x)
        ecdf2 = ecdf(y)
        grid = range(minimum(vcat(x, y)), maximum(vcat(x, y)); length=100)
        dx = step(grid)
        diffs = abs.(ecdf1(grid) .- ecdf2(grid))
        tot += sum(diffs) * dx
    end
    return tot / T
end

"""
    my_loss(tree, dataset, options) -> Float64

Computes a loss value for a given symbolic expression tree representing a drift function,
by comparing simulated trajectories to real data using ECDF-based distance.

Arguments:
    tree: a symbolic expression tree representing the candidate drift function
    dataset: an object with field `X`, a flattened (1, T*N) matrix of real trajectories
    options: a dictionary or object with at least the field `operators` to evaluate the expression tree

Returns:
    A scalar loss value representing the distance between real and simulated trajectories
"""
function my_loss(tree, dataset, options)
    drift_fn = x -> eval_tree_array(tree, reshape(x, 1,:), options.operators)[1]
    T = 100  # This must match the T used in simulation
    N = size(dataset.X, 2) ÷ T
    original_trajectories = unflatten_matrix(dataset.X, N)
    x0 = original_trajectories[1, :]
    simulated_trajectories = simulate(drift_fn, x0)
    return compute_distance(original_trajectories, simulated_trajectories)
end

function loss_2expr(tree, dataset, options)
    # Define drift and diffusion functions from tree
    drift_fn = x -> eval_tree_array(tree.drift, reshape(x, 1, :), options.operators)[1]
    diffusion_fn = x -> eval_tree_array(tree.diffusion, reshape(x, 1, :), options.operators)[1]

    T = 100  # Must match simulation time steps
    N = size(dataset.X, 2) ÷ T
    original_trajectories = unflatten_matrix(dataset.X, N)
    x0 = original_trajectories[1, :]

    # Simulate using both drift and diffusion
    simulated_trajectories = simulate(drift_fn, diffusion_fn, x0)

    # Compare real vs simulated
    return compute_distance(original_trajectories, simulated_trajectories)
end


function plot_trajectories(
    filename::String,
    trajectories::Matrix{Float64},
    dt=0.1)

    T, N = size(trajectories)
    time = collect(0:dt:(T-1)*dt)
    plt = plot(legend = false)
    for i in 1:N
        yi = trajectories[:, i]
        plot!(time, yi)
    end
    xlabel!(plt, "t")
    ylabel!(plt, "y")
    title!(plt, "Trajectories")
    savefig(plt, filename)
end

function wrap_text(s::String, max_length::Int)
    parts = []
    while !isempty(s)
        cut_limit = min(max_length, lastindex(s))
        cut = findlast(isspace, s[1:cut_limit])
        cut = isnothing(cut) ? cut_limit : cut
        push!(parts, strip(s[1:cut]))
        s = s[cut+1:end]
    end
    return join(parts, "\n")
end

function plot_comparison(
    filename::String,
    original::Matrix{Float64},
    predicted::Matrix{Float64},
    dt::Float64,
    distance::Float64,
    original_expr::String,
    predicted_expr::String)

    T, N = size(original)
    time = collect(0:dt:(T - 1) * dt)
    left_title = "ORIGINAL\nDrift: " * wrap_text(original_expr, 70)
    right_title = "PREDICTED (Distance = $(round(distance, digits=4)))\nDrift: " * wrap_text(predicted_expr, 70)
    p1 = plot(title=left_title, titlefontsize=9, xlabel="t", ylabel="y", legend=false)
    for i in 1:N
        plot!(p1, time, original[:, i])
    end
    p2 = plot(title=right_title, titlefontsize=9, xlabel="t", legend=false)
    for i in 1:N
        plot!(p2, time, predicted[:, i])
    end
    plt = plot(p1, p2, layout=(1, 2), size=(1000, 500))
    savefig(plt, filename)
end

struct SDE{T}
    drift::T
    diff::T
end

#output = [SDE(b...) for b in data.B]

function compute((; drift, diff), (y,))
    _f = drift(y)
    _g = diff(y)
    results = [SDE(f, g) for (f, g) in zip(_f.x, _g.x)]

    return ValidVector(results, _f.valid && _g.valid)
end

structure = TemplateStructure{(:drift, :diff,)}(compute)

##### PROGRAM ##########

drift_functions = [
    y -> -0.3 .* y,
    y -> 0.2 .* y,
    y -> -0.2 .* y .+ 1.0,
    y -> 0.2 .* exp.(-y .+ 1.5),
    y -> 0.9 .* sin.(y),
]

drift_exprs = [
    "-0.3 .* y",
    "0.2 .* y",
    "-0.2 .* y .+ 1.0",
    "0.2 .* exp.(-y.+1.5)",
    "0.9 .* sin.(y)",
]

diffusion_functions = [
    y -> 0.2 .+ 0.0 .* y,
    y -> 0.3 .* y,
    y -> 0.4 .* exp.(-0.5 .* y),
    y -> sin.(y),
    y -> 0.1 .* y,
]

diffusion_exprs = [
    "0.2 .+ 0.0 .* y",
    "0.3 .* y",
    "0.4 .* exp.(-0.5 .* y)",
    "sin.(y)",
    "0.1 .* y",
]

N = 50
T = 100
dt = 0.1
x0 = randn(N) # Random Initial Condtions
#=
for (i, drift) in enumerate(drift_functions)
	X = simulate(drift, x0)
	y = randn(N * T) # Not used in regression

	options = SymbolicRegression.Options(
	    binary_operators = [+, *, /, -],
	    unary_operators  = [exp, sin],
	    nested_constraints = [
            sin => [sin => 0, exp => 0],
			exp => [exp => 0, sin => 0],
		],
		complexity_of_operators = [sin => 2, exp => 2],
	    loss_function    = my_loss,
	    populations      = 2,
	    maxsize          = 8,
	)

	hall_of_fame = equation_search(
	    flatten_matrix(X),
	    y,
	    niterations = 5,
	    options = options,
	    parallelism = :multithreading,
	)

	dominating = calculate_pareto_frontier(hall_of_fame)
	best_expression = dominating[end]
	drift_fn = x -> eval_tree_array(best_expression.tree, reshape(x, 1,:), options.operators)[1]
	simulated_trajectories = simulate(drift_fn, x0)
	distance = compute_distance(X, simulated_trajectories)
	plot_comparison("comparison_$(i)", X, simulated_trajectories, dt, distance, drift_exprs[i], string(best_expression.tree))
end
=#

for i in 1:length(drift_functions)
    println("Fitting model for system $i")

    drift = drift_functions[i]
    diff = diffusion_functions[i]

    X = simulate(drift, diff, x0)
    output = zeros(N * T)           # One target per observation
    input = reshape(X, :, 1)        # Each row = one observation, one feature



    # Define the symbolic regression model
    model = SRRegressor(
        binary_operators = (+, -, *, /),
        unary_operators = (sin, cos, sqrt, exp),
        niterations = 5,
        maxsize = 35,
        loss_function_expression = loss_2expr,  # Uses both drift + diffusion
        expression_spec = TemplateExpressionSpec(; structure),
        batching = true,
        batch_size = 30,
    )

    # Wrap in a machine and fit
    mach = machine(model, input, output)
    fit!(mach)

    # Get best expression from the final hall of fame
    hof = fitted_params(mach).hallOfFame
    best_expr = hof[end]

    # Rebuild functions from predicted expressions
    drift_fn = x -> eval_tree_array(best_expr.tree.drift, reshape(x, 1, :), model.operators)[1]
    diff_fn = x -> eval_tree_array(best_expr.tree.diff, reshape(x, 1, :), model.operators)[1]

    # Re-simulate using predicted drift and diffusion
    simulated_trajectories = simulate(drift_fn, diff_fn, x0)

    # Compute ECDF-based distance
    distance = compute_distance(X, simulated_trajectories)

    # Build formatted string for plotting
    predicted_expr = string("Drift: ", best_expr.tree.drift, "\nDiff: ", best_expr.tree.diff)
    ground_truth_expr = "Drift: $(drift_exprs[i])\nDiff: $(diffusion_exprs[i])"

    # Plot and save comparison
    plot_comparison("comparison_$(i)", X, simulated_trajectories, dt, distance, ground_truth_expr, predicted_expr)
end
