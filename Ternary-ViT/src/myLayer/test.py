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
# model = CustomModel()


# 输入维度 (4, 4, 8, 3, 3)
# input_tensor = torch.randn(1, 1, 8, 1, 1)
# print(input_tensor)
# # 获取第一行
# first_row = input_tensor[:, :, 0, :, :]
# expanded_tensor = torch.zeros(1, 1, 8, 8, 1, 1, device=input_tensor.device)

# for i in range(8):
#     # 计算旋转后的索引
#     rotated_index = (i - 1) % 8  # 旋转一位

#     # 将第一行进行旋转并填入扩展的张量中
#     expanded_tensor[:, :, :, i, :, :] = input_tensor.roll(shifts=rotated_index, dims=2)


# 扩展为 (4, 4, 8, 8, 3, 3)
# expanded_tensor = expanded_tensor.unsqueeze(3).expand(-1, -1, -1, 8, -1, -1)

# 检查结果的维度
# print(expanded_tensor)

x = torch.tensor([[-1,-3,-2,-7],[-5,-8,-7,-6],[1,2,3,4],[8,5,7,6],[9,4,7,6],[3,4,5,6],[-4,-3,-5,-6],[-9,-2,-1,-5]])
w = torch.tensor([[5,1,2,3,4,6],[1,5,3,2,6,4],[-1,-2,-3,-4,-5,-6],[-2,-1,-4,-3,-6,-5]])
y = torch.matmul(x,w)
print(y)
