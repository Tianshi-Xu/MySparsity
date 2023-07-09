# export WORLD_SIZE=2
# export MASTER_ADDR=localhost

CUDA_VISIBLE_DEVICES=1 python train.py -c configs/datasets/cifar10.yml \
  /raid/mengli/datasets/ \
  --model convnext_cifar_nano_hnf 

#% CUDA_VISIBLE_DEVICES=2,4 python -m torch.distributed.launch --nproc_per_node=2 \
# CUDA_VISIBLE_DEVICES=2 python train.py \
#   -c configs/quantized/imagenet/mbv2_imagenet_quant_bwn_adam.yml \
#   /raid/mengli/datasets/imagenet/ \
#   --weight-decay 1e-5 \
#   --model mobilenetv2_100 \
#   --aq-enable --aq-bitw 2 \
  # --distributed \
  # --wq-enable --wq-bitw 1 --wq-per-channel \
  # --weight-decay 1e-10 \
  # --resume output/train/20221014-145118-mobilenetv2_100-224/checkpoint-24.pth.tar \

  # -c configs/quantized/imagenet/mbv2_imagenet_quant_adam.yml \

  # --resume output/train/20221013-083814-mobilenetv2_050-224/checkpoint-10.pth.tar \
  # --initial-checkpoint output/train/20221010-224654-mobilenetv2_100-224/model_best.pth.tar \
  # --use-kd --teacher mobilenetv2_100 --quant-teacher \
  # --teacher-checkpoint output/train/20221010-224654-mobilenetv2_100-224/model_best.pth.tar \

  # --resume output/train/20221007-085822-mobilenetv2_dw_100-224/checkpoint-106.pth.tar
  # --teacher-checkpoint pretrained/mbv2_120ep.pth.tar

  # --initial-checkpoint output/train/20221003-143511-mobilenetv2_100-224/model_best.pth.tar \

  # --initial-checkpoint pretrained/mbv2_120ep.pth.tar \

# python train.py \
#   -c configs/quantized/reactnet_cifar10_300epochs_simple.yml \
#   --model cifar10_reactnet /raid/mengli/datasets/ \
  # --aq-enable --aq-bitw 1 --aq-mode LSQ
  # --initial-checkpoint output/train/20220920-153658-cifar10_resnet34-32/model_best.pth.tar
  # --replace-relu \
  # --initial-checkpoint output/train/20220919-094244-cifar10_mobilenetv2_100-32/model_best.pth.tar
  # -c configs/quantized/mbv2_cifar10_300epochs_simple.yml \
  # --wq-enable --wq-bitw 1 \
  # --wq-mode LSQ \
  # --wq-per-channel \
  # --use-kd \
  # --aa rand-m9-mstd0.5-inc1 # \
  # --quant-teacher \
  # --teacher cifar10_mobilenetv2_100 \
  # --teacher-checkpoint output/train/20220917-092256-cifar10_mobilenetv2_100-32/model_best.pth.tar
  # --initial-checkpoint output/train/20220917-092256-cifar10_mobilenetv2_100-32/model_best.pth.tar \
  # --initial-checkpoint output/train/20220915-123827-cifar10_mobilenetv2_100-32/model_best.pth.tar
