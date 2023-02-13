# Train using ausing Flux
using Flux.Data: DataLoader
using Zygote
using LinearAlgebra
using Statistics
using StatsBase
using Random
using YAML
using CUDA
using Clustering
using CairoMakie
using ColorSchemes
using Hungarian

using dd_clustering

push!(LOAD_PATH, "/home/rkube/repos/kstar_ecei_data")
using kstar_ecei_data


struct GroundTruthResult <: ClusteringResult
    assignments::Vector{Int}   # assignments (n)
end
#truth = GroundTruthResult(all_y .+ 1);

"""
    Load data and reshape
"""
shotnr = 26327

# Number of frames in each sample
num_frames = 24
num_classes = 3

#
num_epochs = 100
batch_size = 256

data_norm, tbase_norm = get_shot_data(26327)

# Re-shape data to fit number of samples
# num_samples = size(data_norm)[end] ÷ num_frames
# data_trf = reshape(data_norm[:, :, 1:(num_samples * num_frames)], (24, 8, num_frames, num_samples));
# # Transform into 3d data and scale
# data_trf = Flux.unsqueeze(data_trf, dims=4)
# #data_trf = (data_trf .- minimum(data_trf)) ./ (maximum(data_trf) - minimum(data_trf))
# data_trf = (data_trf .- mean(data_trf)) ./ std(data_trf)




# Alternative idea:
num_samples = size(data_norm)[end]
# Calculate first and second derivative of image time-series manually
data_deriv1 = data_norm[:, :, 3:end] .- data_norm[:, :, 1:end-2];

data_deriv2 = data_norm[:, :, 1:end-2] .- 2f0 * data_norm[:, :, 2:end-1] .+ data_norm[:, :, 3:end];

# Stack data, first, and second derivative
data_trf = zeros(Float32, 24, 8, 3, 1, num_samples-2)
data_trf[:, :, 1, 1, :] = (data_norm[:, :, 2:end-1] .- mean(data_norm)) ./ std(data_norm);
data_trf[:, :, 2, 1, :] = (data_deriv1 .- mean(data_deriv1)) ./ std(data_deriv1);
data_trf[:, :, 3, 1, :] = (data_deriv2 .- mean(data_deriv2)) ./ std(data_deriv2);

f, a, p = hist(data_trf[:], bins=-10.0:0.1:1.0)
save("hist_trf_$(shotnr).png", f)


labels_truth = get_labels(26327)
# Transform labels so that each `num_frames` long sample in data_trf has exactly one label
# This one label is the one that appears most often among the `num_trf`
#labels_truth = reshape(labels_truth[1:(num_samples * num_frames)], (num_frames, num_samples))
#labels_trf = zeros(Int, (1, num_samples))

# Replace the num_frames labels for each sample with the one that occurs most frequently
#for ix_s ∈ 1:num_samples
#    # Find counts of labels [0, 1, 2] in the current batch
#    c = counts(labels_truth[:, ix_s], 0:2)
#    labels_trf[ix_s] = (0:2)[argmax(c)]
#end
#labels_true = labels_trf[1, :] .+ 1;

# For stacking derivatives
labels_true = labels_truth[2:end-1] .+ 1;
assignments_true = GroundTruthResult(labels_true);


# Move data to gpu
data_trf = data_trf |> gpu;
#labels_true = labels_true |> gpu;
loader = DataLoader((data_trf, labels_true), batchsize=batch_size, shuffle=true);


num_fc = 100

model = Chain(Conv((5, 3, 3), 1 => 16, relu),   # 
              BatchNorm(16),
              MaxPool((2, 1, 1)),
              Conv((5, 3, 1), 16 => 32, relu),  # 
              BatchNorm(32),
              #MaxPool((1, 1, 1)),
              Conv((5, 3, 1), 32 => 32, relu),  # 
              BatchNorm(32),
              x -> Flux.flatten(x),
              Dense(2 * 2 * 1 * 32, num_fc, relu),
              #Dense(5 * num_fc, num_fc, relu),
              BatchNorm(num_fc),
              Parallel(vcat, x -> x,  Chain( Dense(num_fc, num_classes), x -> softmax(x))));
model = model |> gpu;


#params_cnn = Flux.params(model[1:8])
#params_fcn = Flux.params(model[9:end])

ps_all = Flux.params(model);
opt_all = ADAM(1e-4)
#opt_cnn = ADAM(1e-4)
#opt_fcn = ADAM(1e-3)
σ = 10.0f0;


all_loss_cs = zeros(length(loader) * num_epochs);
all_loss_simp = zeros(length(loader) * num_epochs);
all_loss_orth = zeros(length(loader) * num_epochs);
NMI_epoch = zeros(num_epochs)

iter = 1;

λ = 1f-4;

for epoch in 1:num_epochs
    for (x,y) in loader
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

            # xyT = y_hidden' * y_hidden
            # x2 = sum(y_hidden.^2, dims=1)
            # distances_squared2 = x2' .- 2xyT + repeat(x2, this_batch)
            Zygote.ignore() do
                global σ = 0.15f0 * median(distances_squared);
            end
            K = exp.(-0.5f0 .* distances_squared / σ / σ)

            # Calculate the matrix M.
            # M_q,i = ||α_q - e_i||^2 with a_q the class probability of a sample and e_i the i-th corner of the simplex
            # use ||α - e||² = ||α||² + ||e||² - <α,e>. Note that ||e||² = 1
            #xyT = A' * CUDA.CuArray(one(zeros(Float32, num_classes, num_classes)))
            #xyT = A' * one(zeros(Float32, num_classes, num_classes))
            #x2 = sum(A.^2, dims=1);
            #y2 = ones(num_classes)' |> gpu;  # This is ||e||² = 1.
            #M = exp.(-x2' .+ 2xyT - repeat(y2, this_batch))' 

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
        Flux.update!(opt, ps_all, grads)
        # Optimize CNN and FCN separately
        #Flux.update!(opt_cnn, params_cnn, grads)
        #Flux.update!(opt_fcn, params_fcn, grads)
        global iter += 1;
    end

    all_probs = model(data_trf)[end-2:end, :] |> cpu;
    labels_pred = [ix[1] for ix in argmax(all_probs, dims=1)][1,:]

    assignments_pred = GroundTruthResult(labels_pred)
    NMI_epoch[epoch] = mutualinfo(assignments_pred, assignments_true)
    @show NMI_epoch[epoch]
end

 

all_probs = model(data_trf)[end-num_classes+1:end, :] |> cpu;

labels_pred = [ix[1] for ix in argmax(all_probs, dims=1)][1,:];
assignments_pred = GroundTruthResult(labels_pred)
nmi = mutualinfo(assignments_true, assignments_pred)


# Scratch pad for cluster accuracy
# https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/
# https://github.com/Gnimuc/Hungarian.jl

# Calculate the confusion matrix
cm = counts(assignments_true, assignments_pred)

# The predicted class labels are arbitrary. To calculate cluster accuracry we need to 
# find the permutation that maximizes the cluster accuracy sum(tr(cm)) / sum(cm)
cm2 = -cm .+ maximum(cm)
matching = Hungarian.munkres(cm2)
ix_perm = [findfirst(Hungarian.munkres(cm2)[i, :].==Hungarian.STAR) for i = 1:3]
cm_perm = cm[:, ix_perm]
cluster_accuracy = sum(tr(cm_perm)) / sum(cm_perm)

# Plot clustered dataset color-coded
# This is basically a vector of colors with same length as total number of samples (unfolded)
colors_true = ColorSchemes.Accent_3[repeat(labels_true, inner=(num_frames, 1))[:, 1]];
colors_pred = ColorSchemes.Accent_3[repeat(labels_pred, inner=(num_frames, 1))[:, 1]];

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
