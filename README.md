# [LeGR: Filter Pruning via Learned Global Ranking](https://arxiv.org/abs/1904.12368)

This is a work that improves our previous NeurIPS workshop paper [Layer-compensated Pruning for Resource-constrained Convolutional Neural Networks](https://arxiv.org/abs/1810.00518), in both further understanding and empirical results.

## Requirements

- PyTorch 1.0.1
- Python 3.5+

## Running LeGR / MorphNet / AMC

The scripts for reproducing the results in Table 1 and Figure 2 are under scripts/

Within each script, there are several commands that run the experiments

P.S. For MorphNet, we search for the trade-off lambda instead of use a large lambda and grow because we find that the growing phase leads to worse results, which is also observed by Wang et al. in their CVPR work [Growing a brain: Fine-tuning by increasing model capacity](https://www.ri.cmu.edu/wp-content/uploads/2017/06/yuxiongw_cvpr17_growingcnn.pdf)


## Visualizing the search progress of affine transformations

![Visualizing the search progress](./legr_mbnetv2_cifar100_flops0.13.gif)

We provide a script to extract the progress (in architectures explored) when learning the affine transformation. For any LeGR script you run, pass the generated output for searching the affine transformation to the following script will generate a visualization of the search progress

For example:

`python utils/plot_search_progress.py log/resnet56_cifar10_flops0.47_transformations_1_output.log resnet56_cifar10_flops0.47_transformations_1.mp4`

The video will be generated at `./resnet56_cifar10_flops0.47_transformations_1.mp4`


## Citation

If you find this repository helpful, please consider citing our work

    @article{chin2018layer,
    title={Layer-compensated pruning for resource-constrained convolutional neural networks},
    author={Chin, Ting-Wu and Zhang, Cha and Marculescu, Diana},
    journal={arXiv preprint arXiv:1810.00518},
    year={2018}
    }

