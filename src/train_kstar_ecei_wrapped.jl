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

# User-defined packages
using dd_clustering
push!(LOAD_PATH, "/home/rkube/repos/kstar_ecei_data")
using kstar_ecei_data

# Instantiate dataset

shotnr = 26512
wrap_frames = 16
trf = GaussianBlur(3)
my_ds = kstar_ecei_3d(shotnr, wrap_frames, trf)


loader = DataLoader(my_ds, batchsize=16, shuffle=true)

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
              BatchNorm(num_fc),
              Parallel(vcat, x -> x,  Chain( Dense(num_fc, num_classes), x -> softmax(x))));

model = model |> gpu;



ps_all = Flux.params(model);
opt_all = AdamW(1e-4, (0.9, 0.999), 1e-4)
σₖ = 10.0f0;

all_loss_cs = zeros(length(loader) * num_epochs);
all_loss_simp = zeros(length(loader) * num_epochs);
all_loss_orth = zeros(length(loader) * num_epochs);
NMI_epoch = zeros(num_epochs)

num_epochs = 20
iter = 1;
λ = 1f-4;

for epoch in 1:num_epochs
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

            # xyT = y_hidden' * y_hidden
            # x2 = sum(y_hidden.^2, dims=1)
            # distances_squared2 = x2' .- 2xyT + repeat(x2, this_batch)
            Zygote.ignore() do
                global σₖ = 0.15f0 * median(distances_squared);
            end
            K = exp.(-0.5f0 .* distances_squared / σₖ / σₖ)

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



# End of file train_kstar_wrapped.jl