import Pkg
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Distances")
using SymbolicRegression
using StatsBase
using Plots
using Distances
import Base: min

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
function simulate(drift_fn::Function, y0::Vector{Float64}, dt=0.1, t_end=10.0):: Matrix{Float64}
    N = length(y0)
    T = Int(t_end/dt)
    trajectories = Array{Float64, 2}(undef, T, N)
    trajectories[1, :] .= y0
    y = copy(y0)
    sqrtdt = sqrt(dt)
    for t in 2:T
        y .= y .+ drift_fn(y) .* dt .+ 0.1 .* randn(N) .* sqrtdt
        y .= clamp.(y, -1e3, 1e3)
        trajectories[t, :] = y
    end
    return trajectories
end

function simulate_2expr(drift_diff_fn::Function, y0::Vector{Float64}, dt=0.1, t_end=10.0)::Matrix{Float64}
    N = length(y0)
    T = Int(t_end / dt)
    trajectories = Array{Float64, 2}(undef, T, N)
    trajectories[1, :] .= y0
    y = copy(y0)
    sqrtdt = sqrt(dt)
    
    for t in 2:T
        drift, diff = drift_diff_fn(y)
        noise = diff .* randn(N) .* sqrtdt
        y .= y .+ drift .* dt .+ noise
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
#=
function unflatten_matrix(flat_M::Matrix{Float64}, N::Int)::Matrix{Float64}
    T = size(flat_M,2) ÷ N
    return reshape(flat_M, T, N)  # (T, N)
end
=# 

function unflatten_matrix(flat_M::AbstractMatrix{Float64}, N::Int)::Matrix{Float64}
    T = size(flat_M, 2) ÷ N
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

function my_loss_2expr(tree, dataset, options)
    drift_diff_fn = x -> begin
        pred, flag = eval_tree_array(tree, reshape(x, 1, :), options.operators)
        return (pred[1].drift, pred[1].diff)
    end
    T = 100  # Must be consistent with t1/dt in simulate
    N = size(dataset.X, 2) ÷ T
    if N != 50
        return 1e10
    end
    original_trajectories = unflatten_matrix(dataset.X, N)
    x0 = original_trajectories[1, :]
    simulated_trajectories = simulate_2expr(drift_diff_fn, x0)
    loss = compute_distance(original_trajectories, simulated_trajectories)
    return loss
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
    original_exprs::NamedTuple,
    predicted_exprs::NamedTuple,
)
    T, N = size(original)
    time = collect(0:dt:(T - 1) * dt)

    left_title = "ORIGINAL\n" *
        "Drift: " * wrap_text(original_exprs.drift, 60) * "\n" *
        "Diffusion: " * wrap_text(original_exprs.diff, 60)

    right_title = "PREDICTED (Distance = $(round(distance, digits=4)))\n" *
        "Drift: " * wrap_text(predicted_exprs.drift, 60) * "\n" *
        "Diffusion: " * wrap_text(predicted_exprs.diff, 60)

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


drift_functions = [
    y -> -0.3 .* y,
    y -> 0.2 .* y,
    y -> -0.2 .* y .+ 1.0,
    y -> 0.2 .* exp.(-y .+ 1.5),
    y -> 0.9 .* y,
]

drift_exprs = [
    "-0.3 .* y",
    "0.2 .* y",
    "-0.2 .* y .+ 1.0",
    "0.2 .* exp.(-y.+1.5)",
    "0.9 .* sin.(y)",
]

diff_functions = [
    y -> 1.0 .* y,
    y -> 0.1 .* y,
    y -> -0.1 .* y .+ 1.0,
    y -> 0.2,
    y -> 0.9 .* y
]

diff_exprs = [
    "1.0 .* y",
    "0.1 .* y",
    "-0.1 .* y + 1.0",
    "0.2",
    "0.9 .* y",
]
N = 50
T = 100
dt = 0.1
x0 = randn(N) 

struct SDE{T}
    drift::T
    diff::T
end

function compute((; drift, diff), (y,))
    _f = drift(y)
    _g = diff(y)
    results = [SDE(f, g) for (f, g) in zip(_f.x, _g.x)]

    return ValidVector(results, _f.valid && _g.valid)
end

structure = TemplateStructure{(:drift, :diff,)}(compute)

for (i, (drift, diff)) in enumerate(zip(drift_functions, diff_functions))
    
    drift_diff_fn = y -> (drift(y), diff(y))
    X = simulate_2expr(drift_diff_fn, x0, dt, T * dt)
    
    flat_X = flatten_matrix(X)
    y_dummy = randn(N * T)

    options = SymbolicRegression.Options(
        binary_operators = [+, *, /, -],
        unary_operators  = [exp, sin],
        loss_function_expression = my_loss_2expr,
        expression_spec = TemplateExpressionSpec(; structure),
        maxsize=8,
    )

    hall_of_fame = equation_search(
        flat_X,
        y_dummy,
        niterations = 2,
        options = options,
    )
    
    dominating = calculate_pareto_frontier(hall_of_fame)
    best = dominating[end]
    predicted_drift_expr = string(best.tree.expr.drift)
    predicted_diff_expr = string(best.tree.expre.diff)
    loss_value = best.loss

    
    plot_comparison(
        "result.png",
        original_trajectories,
        predicted_trajectories,
        dt,
        loss_value,
        (drift = drift_exprs[i], diff = diff_exprs[i]),           # true expressions
        (drift = predicted_drift_expr, diff = predicted_diff_expr)     # predicted expressions
    )
    break

end
