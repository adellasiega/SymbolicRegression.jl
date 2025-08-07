using Pkg
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Distances")

using SymbolicRegression
include("simulate.jl")
include("utils.jl")
include("loss_functions.jl")
include("plotting.jl")

using .Simulate
using .Utils
using .LossFunctions
using .Plotting

drift_functions = [
    y -> -0.5 .* y,
    y -> 0.2 .* y,
    y -> -0.2 .* y .+ 1.0,
    y -> 0.2 .* exp.(-y .+ 1.5),
    y -> 0.9 .* y,
]

drift_exprs = [
    "-0.5 .* y",
    "0.2 .* y",
    "-0.2 .* y .+ 1.0",
    "0.2 .* exp.(-y.+1.5)",
    "0.9 .* y",
]

diff_functions = [
    y -> 0.1,
    y -> 0.1 .* y,
    y -> -0.1 .* y .+ 1.0,
    y -> 0.2,
    y -> 0.9 .* y,
]

diff_exprs = [
    "0.1",
    "0.1 .* y",
    "-0.1 .* y + 1.0",
    "0.2",
    "0.9 .* y",
]

N = 50
T = 100
dt = 0.1
x0 = randn(N)

drift_1 = y -> -0.5 .* y
diff_1 = y -> 0.1 
drift_diff_fn_1 = y -> (drift_1(y), diff_1(y))

drift_2 = y -> -0.0818 .* y
diff_2 = y ->  y 
drift_diff_fn_2 = y -> (drift_2(y), diff_2(y))

X1 = simulate(drift_diff_fn_1, x0, dt, T*dt)
X2 = simulate(drift_diff_fn_2, x0, dt, T*dt)

plot_trajectories("X1", X1)
plot_trajectories("X2", X2)
println("Histogram distance: ", compute_histogram_distance(X1, X2))
println("ECDF distance: ", compute_ecdf_distance(X1, X2))

for (i, (drift, diff)) in enumerate(zip(drift_functions, diff_functions))
    
    drift_diff_fn = y -> (drift(y), diff(y))
    X = simulate(drift_diff_fn, x0, dt, T * dt)
    
    plot_trajectories("simulated_trajectories", X)

    flat_X = flatten_matrix(X)
    y_dummy = randn(N * T)

    options = SymbolicRegression.Options(
        binary_operators = [+, *, /, -],
        unary_operators  = [],
        loss_function_expression = custom_loss,  
        expression_spec = TemplateExpressionSpec(; structure),
        maxsize = 10,
        #nested_constraints = [sin => [sin => 0], exp => [sin => 0]],
        #complexity_of_operators = [sin => 3, exp => 3],
    )

    hall_of_fame = equation_search(
        flat_X, 
        y_dummy,
        niterations=100,
        options=options,
    )
    
    best = calculate_pareto_frontier(hall_of_fame)[end]
    println(best)
    break
end
