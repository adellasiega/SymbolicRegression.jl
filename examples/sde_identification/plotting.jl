module Plotting

export plot_trajectories, plot_comparison

using Plots
using ..Utils: wrap_text

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
