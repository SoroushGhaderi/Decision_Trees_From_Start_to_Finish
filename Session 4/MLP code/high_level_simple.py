import torch

x = torch.randn(10,3)
y = torch.randn(10,2)

fc = torch.nn.Linear(3,2)
print("w = ", fc.weight)
print("b = ", fc.bias)

loss_func = torch.nn.MSELoss()

optimizer = torch.optim.SGD(fc.parameters(), lr=0.01)

y_pred = fc(x)
print(y_pred)

loss_value = loss_func(y_pred, y)
print(loss_value.item())

loss_value.backward()
print(fc.weight.grad)
print(fc.bias.grad)

optimizer.step()
print(fc.weight)

