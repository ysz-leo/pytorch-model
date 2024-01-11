
import torch

input_size = (3, 0, 0)
# input_size = (3, 4, 5)
example_input = torch.randn((1,) + input_size, requires_grad=False)

# new_input_size = list(input_size) + list(input_size)  # 在后面添加 [5, 5]
# new_input_size = tuple(new_input_size)  # 转换回元组
# example_input = torch.randn(new_input_size, requires_grad=False)

print(example_input)


