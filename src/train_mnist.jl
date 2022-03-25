using Flux
using Flux.Data: DataLoader
using CUDA
using Zygote
using LinearAlgebra
using MLDatasets: MNIST
using Plots
using Statistics
using Random

using dd_clustering
ENV["GKSwstype"]="nul"

n_features = 28*28;
num_classes = 10;
num_epochs = 10;
batch_size = 100;

# Get the dataset
train_x, train_y = MNIST.traindata(Float32);
test_x, test_y = MNIST.traindata(Float32);

train_x = 2f0 * reshape(train_x, 28, 28, 1, :) .- 1f0 |>gpu;
test_x = 2f0 * reshape(test_x, 28, 28, 1, :) .- 1f0 |> gpu;
train_y = Flux.onehotbatch(train_y, 0:9) |> gpu;
test_y = Flux.onehotbatch(test_y, 0:9) |> gpu;

loader_train = DataLoader((data=train_x, label=train_y), batchsize=batch_size, shuffle=true);
loader_test = DataLoader((data=train_x, label=train_y), batchsize=batch_size, shuffle=true);

model = Chain(Conv((5, 5), 1 => 32, relu),
              MaxPool((2, 2)),
              Conv((5, 5), 32 => 64, relu),
              MaxPool((2, 2)),
              x -> flatten(x),
              Dense(4 * 4 * 64, 100, relu),
              BatchNorm(100),
              Parallel(vcat, x -> x,  Chain(Dense(100, num_classes, init=Flux.kaiming_uniform), x -> softmax(x)))) |> gpu;

ps = Flux.params(model);
#opt = Flux.Optimiser(ClipValue(1e-3), ADAM(1e-4))
opt = ADAM(1e-3);

sigma2 = 1.0f0;

all_loss_cs = zeros(length(loader_train) * num_epochs);
all_loss_simp = zeros(length(loader_train) * num_epochs);
all_loss_orth = zeros(length(loader_train) * num_epochs);

iter = 1;

for epoch in 1:num_epochs
    for (x,y) in loader_train
        size(x)[end] != batch_size && continue
        loss, back = Zygote.pullback(ps) do 
            y_pred = model(x);
            y_hidden = y_pred[1:end - num_classes, :]       # Output of the last fully-connected layer before the softmax
            A = y_pred[end - num_classes + 1:end, :]            # A contains the cluster assignments. Compared to the article, A is transposed

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
            xyT = A' * CUDA.CuArray(one(randn(Float32, num_classes, num_classes)))
            x2 = sum(A.^2, dims=1);
            y2 = ones(num_classes)' |> gpu;
            M = exp.(-sqrt.(x2' .- 2xyT + repeat(y2, batch_size) .+ eps(eltype(x2))))'

            # Linear algebra below corresponds to ∑_{i=1:N-1}_{j=i+1:N}  A[i, :]' * K * A[j, :] / √(A[i, :]' * K * A[i, :] * A[j, :]' * K * A[j, :]))
            # See https://github.com/DanielTrosten/mvc/blob/b0a08fc6c75bdb1fae796f82a7cbfb001bf02047/src/lib/loss.py#L30
            nom_cs = A * K * A';
            dnom_cs = diag(nom_cs) * diag(nom_cs)';
            loss_cs = 2f0 / (num_classes * (num_classes -1)) * sum(triu(nom_cs ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1))

            nom_simp = M * K * M';
            dnom_simp = diag(nom_simp) * diag(nom_simp)';
            loss_simp =  2f0 / (num_classes * (num_classes -1)) * sum(triu(nom_simp ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1))
            loss_orth = 2f0 * sum(triu(A' * A, 1)) / (batch_size * (batch_size -1))

            @show loss_cs, loss_simp,loss_orth, sigma2
            Zygote.ignore() do 
                all_loss_cs[iter] = loss_cs
                all_loss_simp[iter] = loss_simp 
                all_loss_orth[iter] = loss_orth
            end
            (loss_cs + loss_simp) / num_classes + loss_orth
        end
        grads = back(one(loss))
        Flux.update!(opt, ps, grads)
        iter += 1;
    end

    # Show class assignments in the batch
    #@show sum(model(x)[end-9:end, :], dims=2)'

    p = plot_mnist_classified(model, loader_test, num_classes, epoch);
    savefig(p, "epoch_$(epoch).png")
end

 