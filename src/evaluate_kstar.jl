using CairoMakie
using ColorSchemes
using Clustering
using Hungarianu


"""
    get_labels_predicted(model, x_all)

Returns predicted labels from a trained model.
"""
function get_labels_predicted(model, x_all)
    # Transform entire dataset and predict classes.
    all_probs = model(x_all)[end - num_classes + 1:end, :] |> cpu;

    # Calculate normalized mututal information
    labels_pred = [ix[1] for ix in argmax(all_probs, dims=1)][1,:];
    GroundTruthResult(labels_pred)
end



"""
    get_cluster_accuracy(assignments_true, assignments_pred)

Calculate the cluster accuracy for a set of true and predicted assignments
using the Hungaran algorithm.
"""

function get_cluster_accuracy(assignments_true::GroundTruthResult, assignments_pred::GroundTruthResult)

    # Calculate the confusion matrix
    cm = counts(assignments_true, assignments_pred)

    # The predicted class labels are arbitrary. To calculate cluster accuracry we need to 
    # find the permutation that maximizes the cluster accuracy sum(tr(cm)) / sum(cm)
    cm2 = -cm .+ maximum(cm)
    #matching = Hungarian.munkres(cm2)
    ix_perm = [findfirst(Hungarian.munkres(cm2)[i, :].==Hungarian.STAR) for i = 1:3]
    cm_perm = cm[:, ix_perm]
    sum(tr(cm_perm)) / sum(cm_perm)
end


"""
    Plot single ECEI channel in 2 color codings:
    1. True labels
    2. Predicted labels
"""

function plot_eval_wrapped(model, x_all, assignments_true, epoch)

    # Get predicted assignments
    assignments_pred = get_labels_predicted(model, x_all)
    # Get nmi
    nmi = mutualinfo(assignments_true, assignments_pred)
    # Get cluster accuracy
    cluster_accuracy = get_cluster_accuracy(assignments_true, assignments_pred)

    # Create line coloring vectors for the plot. The assignment are unfolded 
    # by the number of wrapped frames.
    colors_true = repeat(ColorSchemes.Accent_3[assignments_true.assignments], inner=size(x_all)[3]);
    colors_pred = repeat(ColorSchemes.Accent_3[assignments_pred.assignments], inner=size(x_all)[3]);

    # Pick a ECEI channel to use for plotting
    signal = x_all[12, 4, :, :, :] |> cpu;
    signal = reshape(signal, prod(size(signal)));
    # For plotting, shift signal by peak-to-valley
    signal_shift = signal .+  (maximum(signal) - minimum(signal))

    group_color = [PolyElement(color=ColorSchemes.Accent_3[i]) for i ∈ 1:3];
    group_name = ["noise", "filaments", "crash"]

    title_str = "Shot $(shotnr) - Accuracy = " * string(round(cluster_accuracy, digits=3)) * ", NMI = " * string(round(nmi, digits=3))


    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Index", title=title_str)

    leg = Legend(fig, group_color, group_name)
    fig[1, 2] = leg
    lines!(ax, signal, color=colors_true)
    lines!(ax, signal_shift, color=colors_pred)
    #text!(1.0, 0.95, text="True", align=(:left, :bottom), fontsize=16)
    #text!(1.0, 1.15, text="Predicted", aligh=(:left, :top), fontsize=16)

    save("plots/$(shotnr)_epoch" * lpad(string(epoch), 2, '0')* "_pred_vs_true.png", fig)

    return fig
end
