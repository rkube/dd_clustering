using Flux
using Flux.Data: DataLoader
using Zygote


using dd_clustering

# Get the dataset
all_data = gen_circles(1.0, 3.0, 4.0, 10_000);

x1, y1 = all_data[1]
x2, y2 = all_data[2]



