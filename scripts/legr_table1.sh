# ResNet56, 400 epochs follows Discrimination-aware Channel Pruningfor Deep Neural Networks
for i in 1 2 3
do
    python -u legr.py --name resnet56_cifar10_flops0.47_transformations_${i} --dataset torchvision.datasets.CIFAR10 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar10.t7 --rank_type l2_weight --tau_hat 0 --long_ft 60 --prune_away 53 --safeguard 0.1 --batch_size 128 --min_lub --no_val --lr 0.01 > ./log/resnet56_cifar10_flops0.47_transformations_${i}_output.log
    python -u legr.py --name resnet56_cifar10_flops0.47_${i} --dataset torchvision.datasets.CIFAR10 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar10.t7 --rank_type l2_weight --tau_hat 0 --long_ft 400 --prune_away 53 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/resnet56_cifar10_flops0.47_transformations_${i}_ea_min.data > ./log/resnet56_cifar10_flops0.47_${i}_output.log
    python -u legr.py --name resnet56_cifar10_flops0.70_${i} --dataset torchvision.datasets.CIFAR10 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar10.t7 --rank_type l2_weight --tau_hat 0 --long_ft 400 --prune_away 30 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/resnet56_cifar10_flops0.47_transformations_${i}_ea_min.data > ./log/resnet56_cifar10_flops0.70_${i}_output.log
done
