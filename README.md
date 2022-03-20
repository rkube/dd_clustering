This is a basic implementation of (https://arxiv.org/abs/1902.04981)[Deep Divergence-Based Approach to Clustering]. See also (https://github.com/DanielTrosten/mvc)[this implementation].


The code in this repository implements clustering of a circle
dataset using a MLP. Run this test case as
```julia
julia -t src/train_circles.jl
```



Using `num_pts=16384`, `batch_size=256`, `test_split=0.5` for the
data loader as well as `relu` activation functions and `hidden_dim=64` training proceeds as suggested by the animations below

![Points from the dataset with cluster assignment color-coded](https://github.com/rkube/dd_clustering/blob/main/docs/points_epoch.gif)

![Decision boundary of the model](https://github.com/rkube/dd_clustering/blob/main/docs/decision_boundary.gif)

!Cluster assignment in the simplex geometry](https://github.com/rkube/dd_clustering/blob/main/docs/cluster_assignment.gif)