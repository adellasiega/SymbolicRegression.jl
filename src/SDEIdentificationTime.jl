module SDEIdentificationTime

export flatten_matrix_t, unflatten_matrix_t,
       simulate_t, make_drift_diff_fn_t,
       custom_loss_t, compute_self_distance_t,
       compute_wasserstein1d_distance_t

using StatsBase

using ..TemplateExpressionModule:
    TemplateExpression, 
    TemplateStructure, 
    TemplateExpressionSpec, 
    ParamVector, 
    has_params, 
    ValidVector

using Plots
using StatsBase
using DynamicExpressions: eval_tree_array

"""
    flatten_matrix_t(states, times)

Flatten trajectories and times into a 2-row matrix.

- `states`: (M, N)  -> M timesteps, N trajectories
- `times` : length M (shared across all trajectories)

Returns: (2, M*N)
  row 1 -> flattened states
  row 2 -> repeated times
"""
function flatten_matrix_t(states::Matrix{Float64}, times::Vector{Float64})::Matrix{Float64}
    M, N = size(states)
    flat_states = vec(states')              # (M*N), row by row
    flat_times  = repeat(times, N)          # repeat full times for each trajectory
    return vcat(flat_states', flat_times')
end


"""
    unflatten_matrix_t(flat_M, N)

Inverse of `flatten_matrix_t`.

- `flat_M`: (2, M*N)
- `N`: number of trajectories

Returns:
  - `states`: (M, N)
  - `times`: Vector{Float64} of length M
"""
function unflatten_matrix_t(flat_M::AbstractMatrix{Float64}, N::Int)
    flat_states = flat_M[1, :]
    flat_times  = flat_M[2, :]
    M = length(flat_states) ÷ N

    states = reshape(flat_states, N, M)'   # (M, N)
    times  = flat_times[1:M]               # timeline from first trajectory
    return states, times
end


"""
    simulate_t(drift_diff_fn, y0; dt=0.1, t_end=10.0)

Simulate non-autonomous SDE:
  dX_t = f(X_t, t) dt + g(X_t, t) dW_t

- `drift_diff_fn(y, t)` must return (drift, diff)
- `y0`: initial state (Vector{Float64})
- Returns: (2, M*N) flattened matrix
"""
function simulate_t(drift_diff_fn::Function, y0::Vector{Float64}; dt=0.1, t_end=10.0)::Matrix{Float64}
    N = length(y0)
    M = Int(round(t_end/dt)) + 1
    states = Array{Float64}(undef, M, N)
    times = collect(0:dt:t_end)

    states[1, :] .= y0
    y = copy(y0)
    sqrtdt = sqrt(dt)

    for k in 2:M
        t = times[k-1]
        drift, diff = drift_diff_fn(y, t)
        y .= y .+ drift .* dt .+ diff .* randn(N) .* sqrtdt
        y .= clamp.(y, -1e3, 1e3)
        states[k, :] .= y
    end

    return flatten_matrix_t(states, times)
end


"""
    make_drift_diff_fn_t(tree, options)

Build drift/diff evaluation function from symbolic regression tree.
"""
function make_drift_diff_fn_t(tree, options)
    return (x, t) -> begin
        input = reshape(vcat(x, t), 1, :)   # pass state + time
        pred, flag = eval_tree_array(tree, input, options.operators)
        drifts = [p.drift for p in pred]
        diffs  = [p.diff  for p in pred]
        return (drifts, diffs)
    end
end


"""
    compute_wasserstein1d_distance_t(X, Y)

Compute average 1D Wasserstein distance between two sets of trajectories.
Both X and Y are (M, N) state matrices.
"""
function compute_wasserstein1d_distance_t(X::Matrix{Float64}, Y::Matrix{Float64})::Float64
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
        tot += sum(abs.(ecdf1(grid) .- ecdf2(grid))) * dx
    end
    return tot / T
end


"""
    custom_loss_t(tree, dataset, options)

Compute loss by comparing dataset trajectories with simulated ones.
"""
function custom_loss_t(tree, dataset, options)
    N = options.N
    original, times = unflatten_matrix_t(dataset.X, N)
    x0 = original[1, :]

    drift_diff_fn = make_drift_diff_fn_t(tree, options)

    simulated_flat = simulate_t(drift_diff_fn, x0; dt=times[2]-times[1], t_end=times[end])
    simulated, _ = unflatten_matrix_t(simulated_flat, N)

    return compute_wasserstein1d_distance_t(original, simulated)
end


"""
    compute_self_distance_t(drift_diff_fn, y0; dt=0.1, t_end=10.0, n_iterations=100)

Estimate self-distance by comparing multiple simulations with the same parameters.
"""
function compute_self_distance_t(drift_diff_fn::Function, y0::Vector{Float64}; dt=0.1, t_end=10.0, n_iterations=100)
    N = length(y0)
    self_distance = 0.0
    for n in 1:n_iterations
        X1_flat = simulate_t(drift_diff_fn, y0; dt=dt, t_end=t_end)
        X2_flat = simulate_t(drift_diff_fn, y0; dt=dt, t_end=t_end)
        X1, _ = unflatten_matrix_t(X1_flat, N)
        X2, _ = unflatten_matrix_t(X2_flat, N)
        self_distance += compute_wasserstein1d_distance_t(X1, X2)
    end
    return self_distance/n_iterations
end

end # module
