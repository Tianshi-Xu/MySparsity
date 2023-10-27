# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/resnet32/fp32.yml /home/xts/code/dataset/cifar100/ --model resnet32                                          

# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/mbv2/tiny_w8a2.yml /home/xts/code/dataset/tiny/tiny-imagenet-200/ --model tinyimagenet_mobilenetv2 


CUDA_VISIBLE_DEVICES=3 python train.py -c ./configs/quantized/resnet32/w2a3.yml /home/xts/code/dataset/cifar100 --model resnet32
