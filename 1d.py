import torch

print("Empty torch tensor of dimension 4x6")
a = torch.empty(4, 6)
print(a)

print("Random torch tensor of dimension 2x3x1")
b = torch.rand(2, 3, 1)
print(b)

print("Torch tensor with values = 1 of dimension 4")
c = torch.ones(4)
print(c)


print("Example to calculate gradients")
print("x:")
x = torch.ones(3, 3, requires_grad=True)
print(x)

print("y:")
y = x + 2
print(y)

print("z:")
z = y*y*y
print(z)

print("out:")
out = z.mean()
print(out)

print("Backpropagating gradients.")
out.backward()

print("Gradient for x")
print(x.grad)
