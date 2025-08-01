module LossFunctions

export custom_loss
using SymbolicRegression: eval_tree_array
using ..Simulate: simulate 
using ..Utils: unflatten_matrix, compute_distance

function custom_loss(tree, dataset, options)
    drift_diff_fn = x -> begin
        pred, flag = eval_tree_array(tree, reshape(x, 1, :), options.operators)
        return (pred[1].drift, pred[1].diff)
    end
    T = 100
    N = size(dataset.X, 2) ÷ T
    if N != 50
        return 1e10
    end
    original = unflatten_matrix(dataset.X, N)
    x0 = original[1, :]
    simulated = simulate(drift_diff_fn, x0)
    return compute_distance(original, simulated)
end

end
