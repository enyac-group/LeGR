
# MobileNetV2
for i in 1 2 3
do
    for PA in 80 84 87 90
    do
        python -u legr.py --name mbnetv2_cifar100_uniform_${PA}_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerMBNetV2 --model ./ckpt/mbnetv2_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away ${PA} --safeguard 0.1 --batch_size 128 --no_val --lr 0.01 --uniform_pruning > ./log/mbnetv2_cifar100_uniform_0.${PA}_${i}_output.log
    done
done

# ResNet56
for i in 1 2 3
do
    for PA in 82 72 62 52 42 32 22
    do
        python -u legr.py --name resnet56_cifar100_uniform_0.${PA}_${i} --dataset torchvision.datasets.CIFAR100 --datapath ./data --pruner FilterPrunerResNet --model ./ckpt/resnet56_cifar100.t7 --rank_type l2_weight --long_ft 60 --prune_away ${PA} --safeguard 0.1 --batch_size 128 --no_val --uniform_pruning --lr 0.01 > ./log/resnet56_cifar100_uniform_0.${PA}_${i}_output.log
    done
done
