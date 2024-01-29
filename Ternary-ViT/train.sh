# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/resnet32/fp32.yml /home/xts/code/dataset/cifar100/ --model resnet32                                          

# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/mbv2/tiny_fp32.yml /home/xts/code/dataset/tiny-imagenet-200/ --model tinyimagenet_mobilenetv2 


# CUDA_VISIBLE_DEVICES=3 python train.py -c ./configs/quantized/resnet32/fp32.yml /home/xts/code/dataset/cifar100 --model resnet32

CUDA_VISIBLE_DEVICES=0 python train_cirnas.py -c ./configs/datasets/cifar10_kd_nas_mbv2_finetune.yml /home/xts/code/dataset/cifar10 --model cifar_cir_nas_mobilenetv2

# python -m torch.distributed.launch --nproc_per_node=8 train.py -c ./configs/datasets/imagenet_kd.yml /opt/dataset/imagenet --model imagenet_cir_mobilenetv2

# CUDA_VISIBLE_DEVICES=1 python train.py -c ./configs/quantized/resnet18_tiny/w2a2.yml /home/xts/code/dataset/tiny-imagenet-200/ --model tiny_resnet18