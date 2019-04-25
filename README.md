# LeGR: Filter Pruning via Learned Global Ranking

## Requirements

- PyTorch 0.4.0 / 1.0.0
- Python 3.5+

## Running LeGR

To produce the CIFAR-10 results presented in Table 1, please run the following command:

`bash legr_table1.sh`

It will learn the affine transformations and prune accordingly.


## Visualizing the search progress

We provide a script to extract the progress (in architectures explored) when learning the affine transformation.
Assuming you've run the above script, run the following command to generate a video for visualization

`python utils/plot_search_progress.py log/resnet56_cifar10_flops0.47_transformations_1_output.log resnet56_cifar10_flops0.47_transformations_1.mp4`

The video will be generated at `./resnet56_cifar10_flops0.47_transformations_1.mp4`
