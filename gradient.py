import torch

x = torch.tensor(2., requires_grad=True)
y = torch.tensor(3., requires_grad=True)
out = x * y
out.backward()
print(x.grad)
print(y.grad)
# x.grad = 3
# y.grad = 2
out = x * y
out.backward()
print(x.grad)
# x.grad = 6

################################################
x = torch.tensor(2., requires_grad=True)
y = x * x
z = y * y
z.backward()
print(x.grad)
print(y.grad)
