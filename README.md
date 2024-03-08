# MySparsity
#### Requirement
```
torch 2.1.0
when other packages are missing, just pip install them
```
#### Get Start
```
change ./Ternary-ViT/finetune_imagenet.py #line 37 the pytorch-image-models path to your own path 
```

#### Finetune the Block Circulant CNN
```
cd ./Ternary-ViT/
bash train_imagenet.sh
```
Change the config file  in ./configs/datasets/imagenet_finetune.yml
- `Block size`, the block size of circulant matrix, we need to evaluate block size=2/4/8/16
- `lr`, 0.05 is fine, when training with multiple gpus, the `lr` will times by the number of gpus automatically.
- Change `log_name` to your own name, it will generate a log_name.log file.
- Change the `teacher_checkpoint`/`initial_checkpoint` to your path.

Single gpu training:
- In config file, change  `distributed` to false.
```
CUDA_VISIBLE_DEVICES=0 python finetune_imagenet.py -c [your_config_path] [your_dataset_path]  --model finetune_imagenet_cir_nas_mobilenetv2_fix

# For example:
CUDA_VISIBLE_DEVICES=0 python finetune_imagenet.py -c ./configs/datasets/imagenet_finetune.yml /opt/dataset/imagenet --model finetune_imagenet_cir_nas_mobilenetv2_fix

```
Multiple gpus training:
- In config file, change  `distributed` to true. Others are the same as single gpu training.
```
CUDA_VISIBLE_DEVICES=0,1,3,4 python -m torch.distributed.launch --nproc_per_node=4 finetune_imagenet.py -c ./configs/datasets/imagenet_finetune.yml /opt/dataset/imagenet --model finetune_imagenet_cir_nas_mobilenetv2_fix
```
