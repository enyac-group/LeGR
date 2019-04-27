#!/bin/bash

# MobileNetV2
for i in 1 2 3
do
    for PA in 90 87 83 80
    do
        python -u amc.py --name amc_mbnetv2_cifar100_0.${PA}_${i} --datapath ./data --dataset torchvision.datasets.CIFAR100 --model ckpt/mbnetv2_cifar100.t7 --pruner FilterPrunerMBNetV2 --prune_away ${PA} --metric l2_weight --steps 200 --long_ft 60 --max_sparse 90 > ./log/amc_mbnetv2_cifar100_0.${PA}_${i}_output.log
    done
done

# ResNet56
for i in 1 2 3
do
    for PA in 90 87 83 80
    do
        python -u amc.py --name amc_resnet56_cifar100_0.${PA}_${i} --datapath ./data --dataset torchvision.datasets.CIFAR100 --model ckpt/resnet56_cifar100.t7 --pruner FilterPrunerResNet --prune_away ${PA} --metric l2_weight --steps 200 --long_ft 60 --max_sparse 90 > ./log/amc_resnet56_cifar100_0.${PA}_${i}_output.log
    done
done
