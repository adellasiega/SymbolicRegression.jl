module SDEIdentification

export 
    custom_loss, 
    plot_comparison, 
    plot_trajectories, 
    simulate, 
    flatten_matrix,
    unflatten_matrix, 
    compute_kolmogorov_distance,
    compute_wasserstein1d_distance,
    compute_histogram_distance,
    wrap_text,
    SDE,
    compute,
    structure

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

function custom_loss(tree, dataset, options)
    drift_diff_fn = x -> begin
        pred, flag = eval_tree_array(tree, reshape(x, 1, :), options.operators)
        return (pred[1].drift, pred[1].diff)
    end
    T = 100
    N = size(dataset.X, 2) ÷ T
    original = unflatten_matrix(dataset.X, N)
    x0 = original[1, :]
    simulated = simulate(drift_diff_fn, x0)
    return compute_kolmogorov_distance(original, simulated)
end

function simulate(drift_diff_fn::Function, y0::Vector{Float64}, dt=0.1, t_end=10.0)::Matrix{Float64}
    N = length(y0)
    T = Int(t_end / dt)
    trajectories = Array{Float64}(undef, T, N)
    trajectories[1, :] .= y0
    y = copy(y0)
    sqrtdt = sqrt(dt)
    for t in 2:T
        drift, diff = drift_diff_fn(y)
        y .= y .+ drift .* dt .+ diff .* randn(N) .* sqrtdt
        y .= clamp.(y, -1e3, 1e3)
        trajectories[t, :] = y
    end
    return trajectories
end

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

const structure = TemplateStructure{(:drift, :diff)}(compute)

function flatten_matrix(M::Matrix{Float64})::Matrix{Float64}
    return reshape(vec(M)', 1, :)
end

function unflatten_matrix(flat_M::AbstractMatrix{Float64}, N::Int)::Matrix{Float64}
    T = size(flat_M, 2) ÷ N
    return reshape(flat_M, T, N)
end

function compute_wasserstein1d_distance(X::Matrix{Float64}, Y::Matrix{Float64})::Float64
    T = size(X, 1)
    tot = 0.0
    for t in 1:T
        x = X[t, :]
        y = Y[t, :]

        if any(isnan.(x))
            println("Detected NaN in x at positions: ", findall(isnan, x))
            return 1e10
        elseif any(isnan.(y))
            #println("Detected NaN in y at positions: ", findall(isnan, y))
            return 1e10
        elseif any(isinf.(x))
            println("Detected Inf in x at positions: ", findall(isinf, x))
            return 1e10
        elseif any(isinf.(y))
            println("Detected Inf in y at positions: ", findall(isinf, y))
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

function compute_kolmogorov_distance(X::Matrix{Float64}, Y::Matrix{Float64})::Float64
    T = size(X, 1)
    tot = 0.0
    for t in 1:T
        x = X[t, :]
        y = Y[t, :]

        if any(isnan.(x))
            println("Detected NaN in x at positions: ", findall(isnan, x))
            return 1e10
        elseif any(isnan.(y))
            #println("Detected NaN in y at positions: ", findall(isnan, y))
            return 1e10
        elseif any(isinf.(x))
            println("Detected Inf in x at positions: ", findall(isinf, x))
            return 1e10
        elseif any(isinf.(y))
            println("Detected Inf in y at positions: ", findall(isinf, y))
            return 1e10
        end

        ecdf1 = ecdf(x)
        ecdf2 = ecdf(y)
        grid = range(minimum(vcat(x, y)), maximum(vcat(x, y)); length=100)
        tot += max(abs.(ecdf1(grid) .- ecdf2(grid)))
    end
    return tot / T
end


function compute_histogram_distance(X::Matrix{Float64}, Y::Matrix{Float64}; nbins::Int=10)::Float64
    T = size(X, 1)
    N = size(X, 2)
    tot = 0.0

    for t in 1:T
        x = X[t, :]
        y = Y[t, :]

        if any(isnan.(x))
            println("Detected NaN in x at positions: ", findall(isnan, x))
            return 1e10
        elseif any(isnan.(y))
            #println("Detected NaN in y at positions: ", findall(isnan, y))
            return 1e10
        elseif any(isinf.(x))
            println("Detected Inf in x at positions: ", findall(isinf, x))
            return 1e10
        elseif any(isinf.(y))
            println("Detected Inf in y at positions: ", findall(isinf, y))
            return 1e10
        end

        # Shared bin edges
        lo = min(minimum(x), minimum(y))
        hi = max(maximum(x), maximum(y))
        edges = range(lo, hi; length=nbins+1)

        # Normalized histograms
        hist_x = fit(Histogram, x, edges).weights
        hist_y = fit(Histogram, y, edges).weights

        pdf_x = hist_x / sum(hist_x)
        pdf_y = hist_y / sum(hist_y)

        # L1 distance and rescale to [0, 1]
        dist = sum(abs.(pdf_x .- pdf_y)) / 2
        tot += dist
    end

    return tot / T
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

function plot_trajectories(filename::String, trajectories::Matrix{Float64}, dt=0.1)
    T, N = size(trajectories)
    time = collect(0:dt:(T - 1) * dt)
    plt = plot(legend=false)
    for i in 1:N
        plot!(plt, time, trajectories[:, i])
    end
    xlabel!(plt, "t")
    ylabel!(plt, "y")
    title!(plt, "Trajectories")
    savefig(plt, filename)
end

function plot_comparison(filename::String, original::Matrix{Float64}, predicted::Matrix{Float64},
                         dt::Float64, distance::Float64,
                         original_exprs::NamedTuple, predicted_exprs::NamedTuple)

    T, N = size(original)
    time = collect(0:dt:(T - 1) * dt)

    left_title = "ORIGINAL\nDrift: " * wrap_text(original_exprs.drift, 60) *
        "\nDiffusion: " * wrap_text(original_exprs.diff, 60)
    right_title = "PREDICTED (Distance = $(round(distance, digits=4)))\nDrift: " *
        wrap_text(predicted_exprs.drift, 60) * "\nDiffusion: " *
        wrap_text(predicted_exprs.diff, 60)

    p1 = plot(title=left_title, xlabel="t", ylabel="y", legend=false)
    for i in 1:N
        plot!(p1, time, original[:, i])
    end

    p2 = plot(title=right_title, xlabel="t", legend=false)
    for i in 1:N
        plot!(p2, time, predicted[:, i])
    end

    plt = plot(p1, p2, layout=(1, 2), size=(1000, 500))
    savefig(plt, filename)
end


end
