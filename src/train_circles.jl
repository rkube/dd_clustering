using Flux
using Flux.Data: DataLoader
using Zygote
using LinearAlgebra
using Plots
using Statistics
using Random

using dd_clustering


hidden_dim = 128
num_classes = 2

r1 = 0.5
r2 = 1.0
r3 = 2.0
num_pts = 10000;

num_epochs = 20

batch_size = 1024;
test_split = 0.5;

# Get the dataset
X, Y = gen_circles(r1, r2, r3, num_pts);

# Scale X
X = 2.0 * (X .- minimum(X)) / (maximum(X) - minimum(X)) .- 1.0;

num_test = Int(round(num_pts * test_split));
all_idx = shuffle(1:num_pts);
train_idx = all_idx[1:num_test];
test_idx = all_idx[num_test + 1:end];

loader_train = DataLoader((data=X[:, train_idx], label=Y[:, train_idx]), batchsize=batch_size, shuffle=true);
loader_test = DataLoader((data=X[:, test_idx], label=Y[:, test_idx]), batchsize=batch_size, shuffle=true);

model = Chain(Dense(2, hidden_dim, swish, init=Flux.kaiming_uniform),
              Dense(hidden_dim, hidden_dim, swish, init=Flux.kaiming_uniform),
              Dense(hidden_dim, hidden_dim, swish, init=Flux.kaiming_uniform),
              Parallel(vcat, x -> x, Chain(Dense(hidden_dim, num_classes, init=Flux.kaiming_uniform), x -> softmax(x))))
ps = Flux.params(model)

opt = ADAM(1e-5)

xy_grid = hcat(repeat(-1.0:0.01:1.0, outer=201), repeat(-1.0:0.01:1.0, inner=201))';
y_pred = model(xy_grid)[end-1:end,:];
label1 = y_pred[1,:] .> y_pred[2,:];
p = plot(xy_grid[1, label1], xy_grid[2, label1], title="epoch $(epoch)", seriestype=:scatter)
plot!(p, xy_grid[1, .!(label1)], xy_grid[2, .!(label1)], seriestype=:scatter)
savefig(p, "decisionboundary_epoch0.png");
x_t, y_t = first(loader_test);
y_pred = model(x_t)[end-1:end, :]
label1 = y_pred[1,:] .> y_pred[2,:]
p = plot(x_t[1, label1], x_t[2, label1], title="epoch $(epoch)", seriestype=:scatter)
plot!(p, x_t[1, .!(label1)], x_t[2, .!(label1)], seriestype=:scatter)
savefig(p, "points_epoch$(epoch).png")

p = plot(y_pred[1, label1], y_pred[2, label1], title="epoch $(epoch)", seriestype=:scatter)
plot!(p, y_pred[1, .!(label1)], y_pred[2, .!(label1)], title="epoch $(epoch)", seriestype=:scatter)
savefig(p, "cluster_assignment$(epoch).png")

for epoch in 1:num_epochs
    for (x,y) in loader_train
        size(x)[2] != batch_size && continue

        loss, back = Zygote.pullback(ps) do 
            y_pred = model(x);
            y_hidden = y_pred[1:end-2, :]       # Output of the last fully-connected layer before the softmax
            A = y_pred[end-1:end, :]            # A contains the cluster assignments. Compared to the article, A is transposed

            distances = reshape([norm(y_hidden[:, i] - y_hidden[:, j]) for i ∈ 1:batch_size, j ∈ 1:batch_size], (batch_size, batch_size))
            sigma2 = 0.15f0 * mean(distances)

            K = exp.(-0.5f0 .* distances .* distances / sigma2)
            M = reshape([exp(-norm(A[:, q] - ((1:num_classes) .== i))^2) for i ∈ 1:num_classes, q ∈ 1:batch_size], (num_classes, batch_size))

            loss_cs = 0.0f0
            loss_orth = 0.0f0
            for i ∈ 1:(num_classes - 1)
                for j ∈ (i + 1):num_classes
                    loss_cs += A[i, :]' * K * A[j, :] / sqrt(A[i, :]' * K * A[i, :] * A[j, :]' * K * A[j, :] + eps(eltype(K)))
                    loss_orth += M[i, :]' * K * M[j, :] / sqrt(M[i, :]' * K * M[i, :] * M[j, :]' * K * M[j, :] + eps(eltype(K)))
                end
            end

            tmp = A' * A;
            loss_triu = sum([i > j ? tmp[i, j] : 0.0 for i ∈ 1:num_classes, j ∈ 1:num_classes])

            @show loss_cs, loss_orth, loss_triu, sigma2
            
            (loss_cs + loss_orth) / num_classes + loss_triu
        end
        grads = back(one(loss))

        Flux.update!(opt, ps, grads)
    end
    xy_grid = hcat(repeat(-1.0:0.01:1.0, outer=201), repeat(-1.0:0.01:1.0, inner=201))';
    y_pred = model(xy_grid)[end-1:end,:]
    label1 = y_pred[1,:] .> y_pred[2,:]
    p = plot(xy_grid[1, label1], xy_grid[2, label1], title="epoch $(epoch)", seriestype=:scatter)
    plot!(p, xy_grid[1, .!(label1)], xy_grid[2, .!(label1)], seriestype=:scatter)
    savefig(p, "decisionboundary_epoch$(epoch).png")

    x_t, y_t = first(loader_test);
    y_pred = model(x_t)[end-1:end, :]
    label1 = y_pred[1,:] .> y_pred[2,:]
    p = plot(x_t[1, label1], x_t[2, label1], title="epoch $(epoch)", seriestype=:scatter)
    plot!(p, x_t[1, .!(label1)], x_t[2, .!(label1)], seriestype=:scatter)
    savefig(p, "points_epoch$(epoch).png")

    p = plot(y_pred[1, label1], y_pred[2, label1], title="epoch $(epoch)", seriestype=:scatter)
    plot!(p, y_pred[1, .!(label1)], y_pred[2, .!(label1)], title="epoch $(epoch)", seriestype=:scatter)
    savefig(p, "cluster_assignment$(epoch).png")

    @show epoch, Flux.Losses.binarycrossentropy(1.0 .- y_pred, y_t)
end

