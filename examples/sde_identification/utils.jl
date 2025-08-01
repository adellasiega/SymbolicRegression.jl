module Utils

export flatten_matrix, unflatten_matrix, compute_distance, wrap_text

using StatsBase

function flatten_matrix(M::Matrix{Float64})::Matrix{Float64}
    return reshape(vec(M)', 1, :)
end

function unflatten_matrix(flat_M::AbstractMatrix{Float64}, N::Int)::Matrix{Float64}
    T = size(flat_M, 2) ÷ N
    return reshape(flat_M, T, N)
end

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
        tot += sum(abs.(ecdf1(grid) .- ecdf2(grid))) * dx
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
