
export gen_circles

using Distributions
using Flux: onehotbatch

function gen_circles(r1, r2, r3, num_pts)
    # Generates dataset that consists of enclosed circles
    # First circle centered at (0,0) with radius r1
    # Second circle centered at (0,0) with points between r2 and r3
    # num_pts gives the total number of points. They are split 50/50 between both circles

    #1. Generate enough random points
    # Split points 50/50
    # First group: Transform random value to [0;r1], assign
    g1 = rand(num_pts ÷ 2) * r1
    phase1 = rand(num_pts ÷ 2) * 2π

    X = cat([g1 .* cos.(phase1) g1 .* sin.(phase1)], dims=1)

    g2 = rand(num_pts ÷ 2) .* (r3 - r2) .+ r2
    phase2 = rand(num_pts ÷ 2) * 2π

    X = cat([X; g2 .* cos.(phase2) g2 .* sin.(phase2)], dims=2)

    # Add labels
    Y = Flux.onehotbatch([zeros(Int, num_pts ÷ 2); ones(Int, num_pts ÷ 2)], 0:1)



    return X
end
