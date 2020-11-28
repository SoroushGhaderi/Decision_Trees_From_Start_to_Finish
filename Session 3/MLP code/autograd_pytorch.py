import torch

x = torch.tensor(1.0)
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)
print(x, w, b)

y = x * w + 2.5*b
print(y)

y.backward()

print(x.grad)
print(w.grad)
print(b.grad)


a = torch.tensor(5., requires_grad=True)
z = a ** 3
print(z)

z.backward()
print(a.grad)