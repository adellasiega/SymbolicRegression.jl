module SDEIdentificationTime

export flatten_matrix_t, unflatten_matrix_t,
       simulate_t, make_drift_diff_fn_t,
       custom_loss_t, compute_self_distance_t,
       compute_wasserstein1d_distance_t,
       compute_wasserstein1d_distance_t_fast,
       structure_t

using StatsBase
using SymbolicRegression.TemplateExpressionModule:
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
    T, N = size(states)
    flat_states = vec(states)               # (T*N)
    flat_times  = repeat(times, N)          # repeat full times for each trajectory
    return vcat(flat_states', flat_times')
end


"""
    unflatten_matrix_t(flat_M, N)

Inverse of `flatten_matrix_t`.

- `flat_M`: (2, T*N)
- `N`: number of trajectories

Returns:
  - `states`: (T, N)
  - `times`: Vector{Float64} of length M
"""
function unflatten_matrix_t(flat_M::AbstractMatrix{Float64}, N::Int)
    flat_states = flat_M[1, :]
    flat_times  = flat_M[2, :]
    T = length(flat_states) ÷ N

    states = Matrix(reshape(flat_states, T, N))   # (T, N)
    times  = flat_times[1:T]               # timeline from first trajectory
    return states, times
end


"""
    simulate_t(drift_diff_fn, y0; dt=0.1, t_end=5.0)

Simulate non-autonomous SDE:
  dX_t = f(X_t, t) dt + g(X_t, t) dW_t

- `drift_diff_fn(y, t)` must return (drift, diff)
- `y0`: initial state (Vector{Float64})
- Returns states (T, N) and times (T,)
"""
function simulate_t(drift_diff_fn::Function, y0::Vector{Float64}; dt=0.1, t_end=5.0)::Tuple{Matrix{Float64}, Vector{Float64}}
    N = length(y0)
    T = Int(round(t_end/dt))
    states = Array{Float64}(undef, T, N)
    times = collect(0.0:dt:t_end-dt)

    states[1, :] .= y0
    y = copy(y0)
    sqrtdt = sqrt(dt)

    for k in 2:T
        t = times[k]
        drift, diff = drift_diff_fn(y, t)
        y .= y .+ drift .* dt .+ diff .* randn(N) .* sqrtdt
        y .= clamp.(y, -1e3, 1e3)
        states[k, :] .= y
    end

    return states, times
end


"""
    make_drift_diff_fn_t(tree, options)

Build drift/diff evaluation function from symbolic regression tree.
"""
function make_drift_diff_fn_t(tree, options)
    return (x::Vector{Float64}, t::Float64) -> begin
        N = length(x)
        drifts = zeros(N)
        diffs  = zeros(N)

        for i in 1:N
            input = reshape([x[i], t], 2, 1)
            pred, flag = eval_tree_array(tree, input, options.operators)
            if pred === nothing || pred[1] === nothing
                # If evaluation fails, return NaN
                drifts[i] = NaN
                diffs[i]  = NaN
            else
                drifts[i] = pred[1].drift
                diffs[i]  = pred[1].diff
            end
        end

        return drifts, diffs
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

function compute_wasserstein1d_distance_t_fast(X::Matrix{Float64}, Y::Matrix{Float64})::Float64

    T = size(X, 1)
    tot = 0.0
    for t in 1:T
        x = X[t, :]
        y = Y[t, :]

        if any(isnan.(x)) || any(isnan.(y)) || any(isinf.(x)) || any(isinf.(y))
            return 1e10
        end
        
        sort!(x)
        sort!(y)

        tot += mean(abs.(x .- y))
    end
    return tot / T
end



"""
    custom_loss_t(tree, dataset, options)

Compute loss by comparing dataset trajectories with simulated ones.
"""
function custom_loss_t(tree, dataset, options)
    N = 25
    original, times = unflatten_matrix_t(dataset.X, N)
    x0 = original[1, :]
    drift_diff_fn = make_drift_diff_fn_t(tree, options)
    sim_states, sim_times = simulate_t(drift_diff_fn, x0)

    return compute_wasserstein1d_distance_t(Matrix(original), sim_states)
end

# SDE struct

struct SDE_t{T}
    drift::T
    diff::T
end

function compute_t((; drift, diff), (y, t, ))
    _f = drift(y, t)
    _g = diff(y, t)
    results = [SDE_t(f, g) for (f, g) in zip(_f.x, _g.x)]
    
    return ValidVector(results, _f.valid && _g.valid)
end

const structure_t = TemplateStructure{(:drift, :diff)}(compute_t)

end # module
