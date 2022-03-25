using Flux
using Flux.Data: DataLoader
using CUDA
using Zygote
using LinearAlgebra
using MLDatasets: MNIST
#using Plots
using Statistics
using Random

using dd_clustering
ENV["GKSwstype"]="nul"


n_features = 28*28;

num_epochs = 10

batch_size = 256;

# Get the dataset
train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.traindata(Float32);

train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |>gpu;
test_x = 2f0 * reshape(test_x, 28, 28, 1, :) .- 1f0 |> gpu;
train_y = Flux.onehotbatch(train_y, 0:9) |> gpu;
test_y = Flux.onehotbatch(test_y, 0:9) |> gpu;

loader_train = DataLoader((data=train_x, label=train_y), batchsize=batch_size, shuffle=true);
loader_test = DataLoader((data=train_x, label=train_y), batchsize=batch_size, shuffle=true);


# Activation function that work well: relu, relu6
# Activation functions that don't work: tanh, sigmoid
model = Chain(Dense(2, hidden_dim, relu, init=Flux.kaiming_uniform),
              Dense(hidden_dim, hidden_dim, relu, init=Flux.kaiming_uniform),
              Dense(hidden_dim, hidden_dim * 2, relu, init=Flux.kaiming_uniform),
              Parallel(vcat, x -> x, Chain(Dense(hidden_dim * 2, num_classes, init=Flux.kaiming_uniform), x -> softmax(x)))) |> gpu;
ps = Flux.params(model)

#opt = Flux.Optimiser(ClipValue(1e-3), ADAM(1e-4))
opt = ADAM(1e-3)

epoch=0
xy_grid = hcat(repeat(-1.0:0.025:1.0, outer=81), repeat(-1.0:0.025:1.0, inner=81))' |> gpu;
y_pred = model(xy_grid)[end-1:end,:];
#label1 = y_pred[1,:] .> y_pred[2,:];
#p = plot(xy_grid[1, label1], xy_grid[2, label1], title="epoch $(epoch)", seriestype=:scatter)
#plot!(p, xy_grid[1, .!(label1)], xy_grid[2, .!(label1)], seriestype=:scatter)
#savefig(p, "decisionboundary_epoch0.png");
#
#x_t, y_t = first(loader_test);
#y_pred = model(x_t)[end-1:end, :];
#label1 = y_pred[1,:] .> y_pred[2,:];
#p = plot(x_t[1, label1], x_t[2, label1], title="epoch $(epoch)", seriestype=:scatter)
#plot!(p, x_t[1, .!(label1)], x_t[2, .!(label1)], seriestype=:scatter)
#savefig(p, "points_epoch$(epoch).png")
#
#p = plot(y_pred[1, label1], y_pred[2, label1], title="epoch $(epoch)", seriestype=:scatter)
#plot!(p, y_pred[1, .!(label1)], y_pred[2, .!(label1)], title="epoch $(epoch)", seriestype=:scatter)
#savefig(p, "cluster_assignment$(epoch).png")

sigma2 = 1.0f0;

for epoch in 1:num_epochs
    # println("\n\n\n\n\n")
    for (x,y) in loader_train
        size(x)[2] != batch_size && continue
        loss, back = Zygote.pullback(ps) do 
            y_pred = model(x);
            y_hidden = y_pred[1:end-2, :]       # Output of the last fully-connected layer before the softmax
            A = y_pred[end-1:end, :]            # A contains the cluster assignments. Compared to the article, A is transposed

            # See https://github.com/DanielTrosten/mvc/blob/b0a08fc6c75bdb1fae796f82a7cbfb001bf02047/src/lib/kernel.py#L45
            # Note that Julia's batch dimension is 1 (using 1-based index). PyTorch's batch dimension is 0, counted in 0-based indexing.
            xyT = y_hidden' * y_hidden
            x2 = sum(y_hidden.^2, dims=1)
            distances_squared = x2' .- 2xyT + repeat(x2, batch_size)
            Zygote.ignore() do
                global sigma2 = 0.15f0 * median(distances_squared);
            end
            K = exp.(-0.5f0 .* distances_squared / sigma2)

            # Calculate the matrix M.
            xyT = A' * CUDA.CuArray([1f0 0f0; 0f0 1f0]);
            x2 = sum(A.^2, dims=1);
            y2 = ones(num_classes)' |> gpu;
            M = exp.(-sqrt.(x2' .- 2xyT + repeat(y2, batch_size) .+ eps(eltype(x2))))'

            loss_cs = 0.0f0
            loss_simp = 0.0f0
            for i ∈ 1:(num_classes - 1)
                for j ∈ (i + 1):num_classes
                    loss_cs += A[i, :]' * K * A[j, :] / sqrt(A[i, :]' * K * A[i, :] * A[j, :]' * K * A[j, :] + eps(eltype(K)))
                    loss_simp += M[i, :]' * K * M[j, :] / sqrt(M[i, :]' * K * M[i, :] * M[j, :]' * K * M[j, :] + eps(eltype(K)))
                end
            end
            loss_orth = 2f0 * sum(triu(A' * A, 1)) / (batch_size * (batch_size -1))

            # Zygote.ignore() do
            #     println("A:")
            #     display(A)
            #     println("K:")
            #     display(K)
            #     println("M:")
            #     display(M)
            #     println("A'*A:")
            #     display(triu(A'*A))
            # end
            @show loss_cs, loss_simp,loss_orth, sigma2
            (loss_cs + loss_simp) / num_classes + loss_orth
        end
        grads = back(one(loss))

        Flux.update!(opt, ps, grads)
    end
    local xy_grid = hcat(repeat(-1.0:0.025:1.0, outer=81), repeat(-1.0:0.025:1.0, inner=81))' |> gpu;
    local y_pred = model(xy_grid)[end-1:end,:];
    local label1 = y_pred[1,:] .> y_pred[2,:];
    local p = plot(xy_grid[1, label1], xy_grid[2, label1], title="epoch $(epoch)", seriestype=:scatter);
    plot!(p, xy_grid[1, .!(label1)], xy_grid[2, .!(label1)], seriestype=:scatter);
    savefig(p, "decisionboundary_epoch$(epoch).png");

    local x_t, y_t = first(loader_test);
    local y_pred = model(x_t)[end-1:end, :];
    local label1 = y_pred[1,:] .> y_pred[2,:];
    local p = plot(x_t[1, label1], x_t[2, label1], title="epoch $(epoch)", seriestype=:scatter);
    plot!(p, x_t[1, .!(label1)], x_t[2, .!(label1)], seriestype=:scatter);
    savefig(p, "points_epoch$(epoch).png");

    local p = plot(y_pred[1, label1], y_pred[2, label1], title="epoch $(epoch)", seriestype=:scatter);
    plot!(p, y_pred[1, .!(label1)], y_pred[2, .!(label1)], title="epoch $(epoch)", seriestype=:scatter);
    savefig(p, "cluster_assignment$(epoch).png");

    @show epoch, Flux.Losses.binarycrossentropy(1.0 .- y_pred, y_t)
end

