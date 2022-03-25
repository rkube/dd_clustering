export plot_mnist_classified

using Plots

function plot_mnist_classified(model, loader, num_classes, epoch)
    # model - trained classifier
    # loader - fetches examples from the test set 
    # num_classes - number of classes and grid for the output plot

    (x,y) = first(loader);
    y_pred = model(x);
    img_all_classes = zeros(eltype(x), (28*num_classes, 28 * num_classes));
    # Get the class to which each example is assigned to
    class_assignments = map(p -> p[1], argmax(model(x)[end - num_classes + 1:end, :], dims=1));

    for c ∈ 1:num_classes
        # Find where class_assignment is the current class and add the image to the correct row 
        # in img_all_classes
        counter = 1
        for idx ∈ findall(class_assignments .== c)
            counter > 10 && break
            img_all_classes[((c-1) * 28 + 1): (c * 28), ((counter - 1) * 28 + 1):(counter * 28)] = (x[:, end:-1:1, 1, idx[2]]')
            counter += 1
        end
    end
    p = heatmap(img_all_classes, title="epoch $(epoch)", legend=:none, c=cgrad([:black, :white], [-1.0,  1.0]))
end