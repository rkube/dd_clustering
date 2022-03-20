This is a basic implementation of (https://arxiv.org/abs/1902.04981)[Deep Divergence-Based Approach to Clustering].

The code in this repository implements clustering of a circle
dataset using a MLP. Run this test case as
```julia
julia -t src/train_circles.jl
```

Using `num_pts=16384`, `batch_size=256`, `test_split=0.5` for the
data loader as well as `relu` activation functions and `hidden_dim=64` training proceeds as suggested by the animations below

![docs/points_epoch.gif]

![docs/decision_boundary.gif]

![docs/cluster_assignments.gif]