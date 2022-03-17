
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

    rvs = rand(num_pts);
    g1 = rvs[1:num_pts ÷ 2] * r1
    g2 = rvs[num_pts ÷ 2 + 1:end] .* (r3 - r2) .+ r2
    g = [g1; g2]
    phase = rand(num_pts) .* 2π
    X = vcat(g' .* cos.(phase'), g' .* sin.(phase'))

    # Add labels
    Y = Flux.onehotbatch([zeros(Int, num_pts ÷ 2); ones(Int, num_pts ÷ 2)], 0:1)

    return X, Y
end
