using Pkg
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Distances")

using SymbolicRegression

using SymbolicRegression: simulate_t, structure_t, flatten_matrix_t, custom_loss_t, compute_wasserstein1d_distance_t, unflatten_matrix_t, make_drift_diff_fn_t, TemplateExpressionSpec, compute_wasserstein1d_distance_t_fast

#include("SDEIdentificationTime.jl")

using Plots

#using .SDEIdentificationTime

N = 25
T = 50
dt = 0.1
x0 = randn(N)

drift = (y,t) -> -0.3.*y
diff = (y,t) ->  0.01.*t
drift_diff_fn = (y, t) -> (drift(y,t), diff(y,t))

X, times = simulate_t(
    drift_diff_fn,
    x0,
    dt=dt,
    t_end=T*dt
)

flat_X = flatten_matrix_t(X, collect(0.0:dt:(T-1)*dt))
y_dummy = randn(N * T)

function custom_loss_t_fast(tree, dataset, options)
    N = 25
    original, times = unflatten_matrix_t(dataset.X, N)
    x0 = original[1, :]
    drift_diff_fn = make_drift_diff_fn_t(tree, options)
    sim_states, sim_times = simulate_t(drift_diff_fn, x0)

    return compute_wasserstein1d_distance_t_fast(Matrix(original), sim_states)
end
#=
options = SymbolicRegression.Options(
    binary_operators = [+, *, /, -],
    unary_operators  = [sin, exp],
    loss_function_expression = custom_loss_t_fast,
    expression_spec = TemplateExpressionSpec(; structure=structure_t, ),
    maxsize = 16,
    nested_constraints = [
        sin => [sin => 0, exp => 0],
        exp => [sin => 0, exp => 0],
    ],

)

hall_of_fame = equation_search(
    flat_X, 
    y_dummy,
    niterations = 20,
    options=options,
)
=#

drift_pred = (y, t) -> y .* -0.30003
diff_pred = (y,t) ->  -0.0066671
drift_diff_pred_fn = (y,t) -> (drift_pred(y,t), diff_pred(y,t))

X_pred, times = simulate_t(
    drift_diff_pred_fn,
    x0,
)

println("EM distance: ", compute_wasserstein1d_distance_t(X_pred, X))
println("EM distance: ", compute_wasserstein1d_distance_t(X_pred, X))
println("EM distance: ", compute_wasserstein1d_distance_t(X, X_pred))
println("EM distance: ", compute_wasserstein1d_distance_t(X, X_pred))



p = plot(times, X_pred, label="", title="50 Trajectories", xlabel="Time", ylabel="Value")
savefig(p, "trajectories_pred.png")

p = plot(times, X, label="", title="50 Trajectories", xlabel="Time", ylabel="Value")
savefig(p, "trajectories_true.png")

