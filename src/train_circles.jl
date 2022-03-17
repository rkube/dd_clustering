using Flux
using Flux.Data: DataLoader
using Zygote
using LinearAlgebra
using Plots
using Statistics
using Random

using dd_clustering


hidden_dim = 32
num_classes = 2

r1 = 1.0
r2 = 3.0
r3 = 4.0
num_pts = 10_000;

num_epochs = 20

batch_size = 256;
test_split = 0.8;

# Get the dataset
X, Y = gen_circles(r1, r2, r3, num_pts);

# Scale X
X = 2.0 * (X .- minimum(X)) / (maximum(X) - minimum(X)) .- 1.0;

num_test = Int(num_pts * test_split);
all_idx = shuffle(1:num_pts);
train_idx = all_idx[1:num_test];
test_idx = all_idx[num_test + 1:end];

loader_train = DataLoader((data=X[:, train_idx], label=Y[:, train_idx]), batchsize=batch_size, shuffle=true);
loader_test = DataLoader((data=X[:, test_idx], label=Y[:, test_idx]), batchsize=batch_size, shuffle=true);

model = get_simple_mlp(hidden_dim, num_classes)
ps = Flux.params(model)

opt = RMSProp(1e-3)

for epoch in 1:num_epochs
    for (x,y) in loader_train
        size(x)[2] != batch_size && continue

        loss, back = Zygote.pullback(ps) do 
            y_pred = model(x);
            y_hidden = y_pred[1:end-2, :]       # Output of the last fully-connected layer before the softmax
            A = y_pred[end-1:end, :]            # A contains the cluster assignments. Compared to the article, A is transposed

            distances = reshape([norm(y_hidden[:, i] - y_hidden[:, j]) for i ∈ 1:batch_size, j ∈ 1:batch_size], (batch_size, batch_size))
            sigma2 = (0.15 * mean(distances))^2

            K = exp.(-0.5 .* distances .* distances / sigma2)
            M = reshape([exp(-norm(A[:, q] - ((1:num_classes) .== i))^2) for i ∈ 1:num_classes, q ∈ 1:batch_size], (num_classes, batch_size))

            loss_cs = 0.0
            loss_orth = 0.0
            for i ∈ 1:(num_classes - 1)
                for j ∈ (i + 1):num_classes
                    loss_cs += A[i, :]' * K * A[j, :] / sqrt(A[i, :]' * K * A[i, :] * A[j, :]' * K * A[j, :])
                    loss_orth += M[i, :]' * K * M[j, :] / sqrt(M[i, :]' * K * M[i, :] * M[j, :]' * K * M[j, :])
                end
            end

            tmp = A * A';
            loss_triu = sum([i > j ? tmp[i, j] : 0.0 for i ∈ 1:num_classes, j ∈ 1:num_classes])

            # @show loss_cs, loss_orth, loss_triu
            
            (loss_cs + loss_orth) / num_classes + loss_triu
        end
        grads = back(one(loss))

        Flux.update!(opt, ps, grads)
    end
    x_t, y_t = first(loader_test);
    y_pred = model(x_t)[end-1:end, :]
    label1 = y_pred[1,:] .> y_pred[2,:]
    p = plot(x_t[1, label1], x_t[2, label1], seriestype=:scatter)
    plot!(p, x_t[1, .!(label1)], x_t[2, .!(label1)], seriestype=:scatter)

    @show epoch, Flux.Losses.binarycrossentropy(1.0 .- y_pred, y_t)
end

