module Utils

export flatten_matrix, unflatten_matrix, compute_ecdf_distance, compute_histogram_distance, wrap_text

using StatsBase

function flatten_matrix(M::Matrix{Float64})::Matrix{Float64}
    return reshape(vec(M)', 1, :)
end

function unflatten_matrix(flat_M::AbstractMatrix{Float64}, N::Int)::Matrix{Float64}
    T = size(flat_M, 2) ÷ N
    return reshape(flat_M, T, N)
end

function compute_ecdf_distance(X::Matrix{Float64}, Y::Matrix{Float64})::Float64
    T = size(X, 1)
    tot = 0.0
    for t in 1:T
        x = X[t, :]
        y = Y[t, :]
        if any(isnan.(x))
            println("Detected NaN in x at positions: ", findall(isnan, x))
            return 1e10
        elseif any(isnan.(y))
            println("Detected NaN in y at positions: ", findall(isnan, y))
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
            println("Detected NaN in y at positions: ", findall(isnan, y))
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

end
