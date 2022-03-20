
export get_simple_mlp

using Flux

function get_simple_mlp(hidden_dim::Int, num_classes::Int)
    Chain(Dense(2, hidden_dim, relu),
          Dense(hidden_dim, hidden_dim, relu),
          Dense(hidden_dim, hidden_dim, relu),
          Parallel(vcat, x -> x, Chain(Dense(hidden_dim, num_classes), x -> softmax(x))))
end