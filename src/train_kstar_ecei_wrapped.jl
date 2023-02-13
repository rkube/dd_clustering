# Train on frame-wrapped KSTAR ELM dataset
using Augmentor
using CairoMakie
using Clustering
using ColorSchemes
using CUDA
using Flux
using Flux.Data: DataLoader
using Hungarian
using LinearAlgebra
using Random
using Statistics
using StatsBase
using Zygote


struct GroundTruthResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
end

# User-defined packages
using dd_clustering
push!(LOAD_PATH, "/home/rkube/repos/kstar_ecei_data")
using kstar_ecei_data

# Instantiate dataset
shotnr = 26512
wrap_frames = 16

num_epochs = 10     # Training epochs
λ = 1f-1;           # Weighting of orthogonality loss

trf = GaussianBlur(3)
my_ds = kstar_ecei_3d(shotnr, wrap_frames, trf)

# Loads training data
loader = DataLoader(my_ds, batchsize=16, shuffle=true)
# Load all data
loader_all = DataLoader(my_ds, batchsize=size(my_ds.features, 4), shuffle=false)

(x,y) = first(loader)

num_fc = 100
num_classes = 3


model = Chain(Conv((5, 3, 3), 1 => 16, relu),   # 
              BatchNorm(16),
              Conv((5, 3, 3), 16 => 64, relu),  # 
              BatchNorm(64),
              Conv((5, 3, 5), 64 => 64, relu),  # 
              BatchNorm(64),
              Conv((5, 1, 5), 64 => 64, relu),
              BatchNorm(64),                    # 8x2x4x64
              x -> Flux.flatten(x),
              Dense(8 * 2 * 4 * 64, num_fc, relu),
              #Dense(5 * num_fc, num_fc, relu),
              #BatchNorm(num_fc),
              Parallel(vcat, x -> x,  
                       Chain(Dense(num_fc, num_classes), x -> softmax(x))));

model = model |> gpu;

ps_all = Flux.params(model);
opt_all = ADAM(1e-3) # (0.9, 0.999), 1e-4)
σₖ = 10.0f0;

all_loss_cs = zeros((length(loader) + 1) * num_epochs);
all_loss_simp = zeros((length(loader) + 1) * num_epochs);
all_loss_orth = zeros((length(loader) + 1) * num_epochs);
NMI_epoch = zeros(num_epochs);

iter = 1;

for epoch in 1:2
    @show epoch
    for (x,y) in loader
        # Move batch to GPU (apply transformation happens lazily on CPU)
        x = Flux.unsqueeze(x, 4) |> gpu;
        this_batch = size(x)[end]
        loss, back = Zygote.pullback(ps_all) do 
            y_pred = model(x);
            y_hidden = y_pred[1:end - num_classes, :]       # Output of the last fully-connected layer before the softmax
            A = y_pred[end - num_classes + 1:end, :]            # A contains the cluster assignments. Compared to the article, A is transposed

            # See https://github.com/DanielTrosten/mvc/blob/b0a08fc6c75bdb1fae796f82a7cbfb001bf02047/src/lib/kernel.py#L45
            # Note that Julia's batch dimension is 1 (using 1-based index). PyTorch's batch dimension is 0, counted in 0-based indexing.

            # Calculate K with kᵢⱼ = ||xᵢ - xⱼ||²
            # using ||xᵢ - xⱼ||² = ||xᵢ||² + ||xⱼ||² - 2<xᵢ, xⱼ>
            G = y_hidden' * y_hidden
            g = diag(G)
            distances_squared = repeat(g, 1, this_batch) - 2f0 * G + repeat(g', this_batch, 1)

            # distances_squared2 = x2' .- 2xyT + repeat(x2, this_batch)
            Zygote.ignore() do
                global σₖ = max(0.15f0 * median(distances_squared), 1f-2);
            end
            K = exp.(-0.5f0 .* distances_squared / σₖ / σₖ)

            α2 = sum(A.^2, dims=1)       # = ||α||²
            # Note that <α,e> === A
            M = exp.(-α2' .+ A' .- 1f0)'

            # Linear algebra below corresponds to ∑_{i=1:N-1}_{j=i+1:N}  A[i, :]' * K * A[j, :] / √(A[i, :]' * K * A[i, :] * A[j, :]' * K * A[j, :]))
            # See https://github.com/DanielTrosten/mvc/blob/b0a08fc6c75bdb1fae796f82a7cbfb001bf02047/src/lib/loss.py#L30
            nom_cs = A * K * A';
            dnom_cs = diag(nom_cs) * diag(nom_cs)';
            #loss_cs = 2f0 / (num_classes * (num_classes - 1)) * sum(triu(nom_cs ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1))
            loss_cs = sum(triu(nom_cs ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1)) / num_classes

            nom_simp = M * K * M';
            dnom_simp = diag(nom_simp) * diag(nom_simp)';
            #loss_simp =  2f0 / (num_classes * (num_classes - 1)) * sum(triu(nom_simp ./ sqrt.(dnom_cs .+ eps(eltype(dnom_cs))), 1))
            loss_simp = sum(triu(nom_simp ./ sqrt.(dnom_simp .+ eps(eltype(dnom_simp))), 1)) / num_classes
            
            # Enforces balanced class distribution
            #loss_orth = 2f0 * sum(triu(A' * A, 1)) / (this_batch * (this_batch -1))
            loss_orth = sum(triu(A' * A, 1))

            #@show loss_cs, loss_simp,loss_orth, σ
            Zygote.ignore() do 
                all_loss_cs[iter] = loss_cs
                all_loss_simp[iter] = loss_simp 
                all_loss_orth[iter] = loss_orth
            end
            loss_cs + loss_simp + λ * loss_orth
        end
        grads = back(one(loss))
        Flux.update!(opt_all, ps_all, grads)
        # Optimize CNN and FCN separately
        #Flux.update!(opt_cnn, params_cnn, grads)
        #Flux.update!(opt_fcn, params_fcn, grads)
        global iter += 1;
    end

    # all_probs = model(data_trf)[end-2:end, :] |> cpu;
    # labels_pred = [ix[1] for ix in argmax(all_probs, dims=1)][1,:]

    # assignments_pred = GroundTruthResult(labels_pred)
    # NMI_epoch[epoch] = mutualinfo(assignments_pred, assignments_true)
    # @show NMI_epoch[epoch]
end


# Transform entire dataset and predict classes
(x_all, labels_true) = first(loader_all)
assignments_true = GroundTruthResult(labels_true .+ 1)

# Calculate assignments predicted by model
x_all = Flux.unsqueeze(x_all, 4) |> gpu;
all_probs = model(x_all)[end - num_classes + 1:end, :] |> cpu;

# Calculate normalized mututal information
labels_pred = [ix[1] for ix in argmax(all_probs, dims=1)][1,:];
assignments_pred = GroundTruthResult(labels_pred)
nmi = mutualinfo(assignments_true, assignments_pred)

# Plot clustered dataset
colors_true = ColorSchemes.Accent_3[labels_true .+ 1];
colors_pred = ColorSchemes.Accent_3[labels_pred];


group_color = [PolyElement(color=ColorSchemes.Accent_3[i]) for i ∈ 1:3];
group_name = ["noise", "filaments", "crash"]

title_str = "Shot $(shotnr) - Accuracy = " * string(round(cluster_accuracy, digits=3)) * ", NMI = " * string(round(nmi, digits=3))


fig = Figure()
ax = Axis(fig[1, 1], xlabel="Index", title=title_str)
ylims!(ax, 0.9, 1.2)

leg = Legend(fig, group_color, group_name)
fig[1, 2] = leg
lines!(ax, ones(length(colors_true)), linewidth=20, color=colors_true)
lines!(ax, 1.1 * ones(length(colors_pred)), linewidth=20, color=colors_pred)
text!(1.0, 0.95, text="True", align=(:left, :bottom), fontsize=16)
text!(1.0, 1.15, text="Predicted", aligh=(:left, :top), fontsize=16)

save("$(shotnr)_pred_vs_true.png", fig)
# End of file train_kstar_wrapped.jl