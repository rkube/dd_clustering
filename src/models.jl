
export get_simple_mlp

using Flux

function get_simple_mlp(hidden_dim::Int, num_classes::Int)
    Chain(Dense(2, hidden_dim, tanh),
          Dense(hidden_dim, hidden_dim, tanh),
          Dense(hidden_dim, 8, tanh),
          Parallel(vcat, x -> x, Chain(Dense(8, num_classes), x -> softmax(x))))
end