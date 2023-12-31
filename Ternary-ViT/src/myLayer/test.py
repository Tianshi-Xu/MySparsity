import torch
import torch.nn as nn

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        # 定义模型结构
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)

    def __str__(self):
        # 在__str__中添加其他信息
        additional_info = "This is a custom model."
        return super(CustomModel, self).__str__() + "\n" + additional_info

# 创建自定义模型
model = CustomModel()

# 打印模型
print(model)
