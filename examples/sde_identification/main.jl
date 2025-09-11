using Pkg
Pkg.add("Plots")
Pkg.add("StatsBase")
Pkg.add("Distances")

using SymbolicRegression
include("SDEIdentificationTime.jl")

using .SDEIdentificationTime

N = 50
T = 100
dt = 0.1
x0 = randn(N)

drift = (y,t) -> -0.5 .* y .+ 0.3 .* t
diff = (y,t) -> 0.1 
drift_diff_fn = (y,t) -> (drift(y,t), diff(y,t))

X, times = simulate_t(drift_diff_fn, x0, dt=dt, t_end=T*dt)

flat_X = flatten_matrix_t(X, collect(0.0:dt:(T-1)*dt))

y_dummy = randn(N * T)

options = SymbolicRegression.Options(
    binary_operators = [+, *, /, -],
    unary_operators  = [sin, exp],
    loss_function_expression = custom_loss_t,  
    expression_spec = TemplateExpressionSpec(; structure,),
    maxsize = 18,
    nested_constraints = [sin => [sin => 0, exp => 0], exp => [sin => 0, exp => 0]],
)

hall_of_fame = equation_search(
    flat_X, 
    y_dummy,
    niterations = 200,
    options=options,
)
