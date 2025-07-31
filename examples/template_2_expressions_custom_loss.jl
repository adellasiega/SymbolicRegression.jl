using Pkg
Pkg.instantiate()
Pkg.add("MLJBase")
using SymbolicRegression
using SymbolicRegression: ValidVector
using Random
using MLJBase: machine, fit!, predict, report

n = 1000
rng = Random.MersenneTwister(0);
y = 10 .* rand(rng, n)

B = [(sin(yi), exp(-yi / 10)) for yi in y]
data = (; y, B)
input = (;
    y=data.y,
)

struct SDE{T}
    drift::T
    diff::T
end

output = [SDE(b...) for b in data.B]

function compute((; drift, diff), (y,))
    _f = drift(y)
    _g = diff(y)
    results = [SDE(f, g) for (f, g) in zip(_f.x, _g.x)]

    return ValidVector(results, _f.valid && _g.valid)
end

structure = TemplateStructure{(:drift, :diff,)}(compute)

function my_loss(tree, dataset::Dataset{T,L}, options)::L where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    if !flag
        return L(Inf)
    end

    # Compute squared error for each field
    err = zero(L)
    for (pred, trueval) in zip(prediction, dataset.y)
        err += (pred.drift - trueval.drift)^2 + (pred.diff - trueval.diff)^2
    end

    return err / dataset.n
end

model = SRRegressor(;
    binary_operators=(+, -, *, /),
    unary_operators=(sin, cos, sqrt, exp),
    niterations=20,
    maxsize=5,
    expression_spec=TemplateExpressionSpec(; structure),
    ## Note that the elementwise loss needs to operate directly on each row of `y`:
    loss_function_expression = my_loss,
);

#=
Note how we also have to define the custom `elementwise_loss`
function. This is because our `combine_vectors` function
returns a `SDE` struct, so we need to combine it against the truth!
Next, we can set up our machine and fit:
=#

mach = machine(model, input, output)
fit!(mach)
