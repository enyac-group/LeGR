#!/bin/bash

PA=99
# MobileNetV2
for i in 1 2 3
do
    for lambda in 7e-9 1e-8 2e-8
    do
        python -u morphnet.py --name morphnet_mbnetv2_cifar100_l${lambda}_0.${PA}_${i} --datapath ./data --epoch 60 --lr 1e-2 --dataset torchvision.datasets.CIFAR100 --model ./ckpt/mbnetv2_cifar100.t7 --lbda ${lambda} --pruner FilterPrunerMBNetV2 --constraint flops --prune_away 0.${PA} --no_grow > ./log/morphnet_mbnetv2_cifar100_l${lambda}_0.${PA}_${i}_output.log
    done
done

# ResNet56
for i in 1 2 3
do
    for lambda in 9e-9 1e-8 1.1e-8 1.2e-8 1.3e-8
    do
        python -u morphnet.py --name morphnet_resnet56_cifar100_l${lambda}_0.${PA}_${i} --datapath ./data --epoch 60 --lr 1e-2 --dataset torchvision.datasets.CIFAR100 --model ./ckpt/resnet56_cifar100.t7 --lbda ${lambda} --pruner FilterPrunerResNet --constraint flops --prune_away 0.${PA} --no_grow > ./log/morphnet_resnet56_cifar100_l${lambda}_0.${PA}_${i}_output.log

    done
done

