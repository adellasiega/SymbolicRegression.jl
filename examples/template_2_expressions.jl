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

model = SRRegressor(;
    binary_operators=(+, -, *, /),
    unary_operators=(sin, cos, sqrt, exp),
    niterations=5,
    maxsize=35,
    expression_spec=TemplateExpressionSpec(; structure),
    ## Note that the elementwise loss needs to operate directly on each row of `y`:
    elementwise_loss=(F1, F2) -> (F1.drift - F2.drift)^2 + (F1.diff - F2.diff)^2,
    batching=true,
    batch_size=30,
);

#=
Note how we also have to define the custom `elementwise_loss`
function. This is because our `combine_vectors` function
returns a `SDE` struct, so we need to combine it against the truth!
Next, we can set up our machine and fit:
=#

mach = machine(model, input, output)
fit!(mach)
report(mach)
