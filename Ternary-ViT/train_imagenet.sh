
# pretrain imagenet
# CUDA_VISIBLE_DEVICES=3 python finetune_imagenet.py -c ./configs/datasets/imagenet_finetune.yml /opt/dataset/imagenet --model finetune_imagenet_cir_nas_mobilenetv2_fix
CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.launch --nproc_per_node=4 finetune_imagenet.py -c ./configs/datasets/imagenet_finetune.yml /opt/dataset/imagenet --model finetune_imagenet_cir_nas_mobilenetv2_fix
