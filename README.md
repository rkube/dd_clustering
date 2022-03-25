This is a basic implementation of [Deep Divergence-Based Approach to Clustering](https://arxiv.org/abs/1902.04981).
Much of the implementation details follow [this implementation](https://github.com/DanielTrosten/mvc).


The code in this repository implements clustering of a circle
dataset using a MLP. Run this test case as
```julia
julia -t src/train_circles.jl
```

Using `num_pts=16384`, `batch_size=256`, `test_split=0.5` for the
data loader as well as `relu` activation functions and `hidden_dim=64` training proceeds as suggested by the animations below

![Points from the dataset with cluster assignment color-coded](https://github.com/rkube/dd_clustering/blob/main/docs/points_epoch.gif)

![Decision boundary of the model](https://github.com/rkube/dd_clustering/blob/main/docs/decision_boundary.gif)

![Cluster assignment in the simplex geometry](https://github.com/rkube/dd_clustering/blob/main/docs/cluster_assignment.gif)


Clustering of the MNIST dataset was performed using the same hyperparameters reported in the article:
`batch_size = 100`, learning rate was set to `0.001` using an ADAM optimizer. The architecture consists of 2 convolutional
layers, the first one uses 32  `5x5` and the second one 64 `5x5` filters. Each conv layer is followed by a `ReLU` activation
function as well as  a `2x2` max-pool layer. The output of the final conv layer is fed into a fully connected layer with
100 nodes and a `ReLU` activation function. BatchNorm is applied to this layer. The output is concatenated with a
final 10-wide layer with softmax applied to it.

![MNIST clustering](https://github.com/rkube/dd_clustering/blob/main/docs/ddc_mnist.gif)
