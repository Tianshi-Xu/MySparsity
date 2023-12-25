# CUDA_VISIBLE_DEVICES=0 python test.py -c ./configs/quantized/resnet18/w2a3test.yml /home/xts/code/dataset/cifar100/ --model cifar10_resnet18                                  

# CUDA_VISIBLE_DEVICES=2 python test.py -c ./configs/quantized/resnet18_tiny/w4a4test.yml /home/xts/code/dataset/tiny-imagenet-200/ --model tinyimagenet_mobilenetv2 

CUDA_VISIBLE_DEVICES=4 python test.py -c ./configs/quantized/resnet18_tiny/w2a3test.yml /home/xts/code/dataset/tiny-imagenet-200/ --model tiny_resnet18  