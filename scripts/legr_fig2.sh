# MobileNetV2
for i in 1 2 3
do
    # Learning affine transformations
    python -u legr.py --name mbnetv2_cifar100_flops0.13_transformations_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerMBNetV2 --model ./ckpt/mbnetv2_cifar100.t7 --rank_type l2_weight --tau_hat 200 --prune_away 87 --safeguard 0.1 --batch_size 128 --min_lub --no_val --lr 0.01 > ./log/mbnetv2_cifar100_flops0.13_transformations_${i}_output.log

    # Learning resource-constrained CNNs
    python -u legr.py --name mbnetv2_cifar100_flops0.10_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerMBNetV2 --model ./ckpt/mbnetv2_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 90 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/mbnetv2_cifar100_flops0.13_transformations_${i}_ea_min.data > ./log/mbnetv2_cifar100_flops0.10_${i}_output.log
    python -u legr.py --name mbnetv2_cifar100_flops0.13_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerMBNetV2 --model ./ckpt/mbnetv2_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 87 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/mbnetv2_cifar100_flops0.13_transformations_${i}_ea_min.data > ./log/mbnetv2_cifar100_flops0.13_${i}_output.log
    python -u legr.py --name mbnetv2_cifar100_flops0.16_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerMBNetV2 --model ./ckpt/mbnetv2_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 84 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/mbnetv2_cifar100_flops0.13_transformations_${i}_ea_min.data > ./log/mbnetv2_cifar100_flops0.16_${i}_output.log
    python -u legr.py --name mbnetv2_cifar100_flops0.20_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerMBNetV2 --model ./ckpt/mbnetv2_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 80 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/mbnetv2_cifar100_flops0.13_transformations_${i}_ea_min.data > ./log/mbnetv2_cifar100_flops0.20_${i}_output.log
done

# ResNet56
for i in 1 2 3
do
    # Learning affine transformations
    python -u legr.py --name resnet56_cifar100_flops0.20_transformations_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar100.t7 --rank_type l2_weight --tau_hat 200 --prune_away 80 --safeguard 0.1 --batch_size 128 --min_lub --no_val --lr 0.01 > ./log/resnet56_cifar100_flops0.20_transformations_${i}_output.log

    # Learning resource-constrained CNNs
    python -u legr.py --name resnet56_cifar100_flops0.10_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 90 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/resnet56_cifar100_flops0.20_transformations_${i}_ea_min.data > ./log/resnet56_cifar100_flops0.10_${i}_output.log
    python -u legr.py --name resnet56_cifar100_flops0.13_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 87 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/resnet56_cifar100_flops0.20_transformations_${i}_ea_min.data > ./log/resnet56_cifar100_flops0.13_${i}_output.log
    python -u legr.py --name resnet56_cifar100_flops0.16_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 84 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/resnet56_cifar100_flops0.20_transformations_${i}_ea_min.data > ./log/resnet56_cifar100_flops0.16_${i}_output.log
    python -u legr.py --name resnet56_cifar100_flops0.20_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away 80 --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --lub ./log/resnet56_cifar100_flops0.20_transformations_${i}_ea_min.data > ./log/resnet56_cifar100_flops0.20_${i}_output.log
done
