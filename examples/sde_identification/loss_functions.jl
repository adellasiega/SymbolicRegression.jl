module LossFunctions

export custom_loss
using SymbolicRegression: eval_tree_array
using ..Simulate: simulate 
using ..Utils: unflatten_matrix, compute_ecdf_distance, compute_histogram_distance

function custom_loss(tree, dataset, options)
    drift_diff_fn = x -> begin
        pred, flag = eval_tree_array(tree, reshape(x, 1, :), options.operators)
        return (pred[1].drift, pred[1].diff)
    end
    T = 100
    N = size(dataset.X, 2) ÷ T
    original = unflatten_matrix(dataset.X, N)
    x0 = original[1, :]
    simulated = simulate(drift_diff_fn, x0)
    return compute_histogram_distance(original, simulated)
end

end
