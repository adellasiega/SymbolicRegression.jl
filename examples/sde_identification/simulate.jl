module Simulate

export simulate, SDE, compute, structure

using SymbolicRegression: ValidVector, TemplateStructure

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

end
