using Pkg
Pkg.instantiate()
using SymbolicRegression
using SymbolicRegression: ValidVector, Dataset, compute_complexity, string_tree
using Random

n = 1000
rng = Random.MersenneTwister(0);

X = (rand(rng, n) .- 0.5) .* 2.0 
Y = [(2.0* x -20.0, exp(-x / 10.0) + 10.0) for x in X]

struct SDE{T}
    drift::T
    diff::T
end

output = [SDE(y...) for y in Y]
dataset = Dataset(X', output)

function compute((; drift, diff), (x,))
    _f = drift(x)
    _g = diff(x)
    results = [SDE(f, g) for (f, g) in zip(_f.x, _g.x)]

    return ValidVector(results, _f.valid && _g.valid)
end

structure = TemplateStructure{(:drift, :diff,)}(compute)

options = Options(
    binary_operators=[+, *, /, -],
    unary_operators=[sin, exp],
    maxsize=20,
    expression_spec=TemplateExpressionSpec(; structure),
    elementwise_loss=(F1, F2) -> (F1.drift - F2.drift)^2 + (F1.diff - F2.diff)^2,
)

hall_of_fame = equation_search(
    dataset,  
    niterations=40,
    options=options,
    parallelism=:multithreading
)

dominating = calculate_pareto_frontier(hall_of_fame)

println("Complexity\tLoss\tEquation")
for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)
    println("$(complexity)\t$(loss)\t$(string)")
end
