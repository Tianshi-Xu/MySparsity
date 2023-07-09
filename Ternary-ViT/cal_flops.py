import torchvision.models as models
import torch
from ptflops import get_model_complexity_info

def cal(net):
  with torch.cuda.device(0):
    macs, params = get_model_complexity_info(net, (3, 32, 32), as_strings=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

