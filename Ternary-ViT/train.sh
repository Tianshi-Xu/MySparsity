# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/resnet32/fp32.yml /home/xts/code/dataset/cifar100/ --model resnet32                                          

<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=0 python train.py -c ./configs/quantized/mbv2/tiny_w8a2.yml /home/xts/code/dataset/tiny/tiny-imagenet-200/ --model tinyimagenet_mobilenetv2 


CUDA_VISIBLE_DEVICES=3 python train.py -c ./configs/quantized/resnet32/w2a3.yml /home/xts/code/dataset/cifar100 --model resnet32
=======
CUDA_VISIBLE_DEVICES=1 python train.py -c ./configs/quantized/mbv2/tiny_w8a3all.yml /home/xts/code/dataset/tiny/tiny-imagenet-200/ --model tinyimagenet_mobilenetv2  
>>>>>>> 70ea5d09b69c990084d048caa58c5a4b149b0fe2
