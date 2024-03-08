# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/resnet32/fp32.yml /home/xts/code/dataset/cifar100/ --model resnet32                                          

# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/mbv2/tiny_fp32.yml /home/xts/code/dataset/tiny-imagenet-200/ --model tinyimagenet_mobilenetv2 


# CUDA_VISIBLE_DEVICES=3 python train.py -c ./configs/quantized/resnet32/fp32.yml /home/xts/code/dataset/cifar100 --model resnet32

# pretrain
# CUDA_VISIBLE_DEVICES=4 python train_cirnas.py -c ./configs/datasets/cifar10_kd_nas_mbv2_pretrain.yml /home/xts/code/dataset/cifar10 --model pretrain_cifar_cir_nas_mobilenetv2

# normal
CUDA_VISIBLE_DEVICES=5 python train_cirnas.py -c ./configs/datasets/cifar10_kd_nas_mbv2_normal.yml /home/xts/code/dataset/cifar10 --model cifar_cir_nas_mobilenetv2

# finetune
# CUDA_VISIBLE_DEVICES=7 python finetune_cirnas.py -c ./configs/datasets/cifar10_kd_nas_mbv2_finetune.yml /home/xts/code/dataset/cifar10 --model finetune_cifar_cir_nas_mobilenetv2

# pretrain imagenet
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 train_cirnas.py -c ./configs/datasets/imagenet_pretrain.yml /opt/dataset/imagenet --model pretrain_imagenet_cir_nas_mobilenetv2
# CUDA_VISIBLE_DEVICES=0 python train_cirnas.py -c ./configs/datasets/imagenet_pretrain.yml /opt/dataset/imagenet --model pretrain_imagenet_cir_nas_mobilenetv2

# CUDA_VISIBLE_DEVICES=1 python train.py -c ./configs/quantized/resnet18_tiny/w2a2.yml /home/xts/code/dataset/tiny-imagenet-200/ --model tiny_resnet18
