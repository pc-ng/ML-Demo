import numpy as np 
import matplotlib.pyplot as plt 

# problem setup
w = -7
x = np.arange(-20, 21)
y = w*x
m = x.size

plt.figure(1)
plt.plot(x, y, 'wo', label='Problem')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()

# hypothesis function
def h(w):
    return w*x

# loss function
def L(w, m, y):
    return 1/(2*m) * np.sum((h(w)-y)**2)

# w_tilde = 1
# y_tilde = h(w_tilde)

# plt.figure(1)
# plt.plot(x, y_tilde, '-b', linewidth =2, label='hypothesis')
# plt.legend()

loss = L(w_tilde, m, y)
print('loss: ' + str(loss))

# visualize the loss for some possible values of w
some_w = np.arange(-50, 50)
some_loss = np.zeros((some_w.size, 1))
for i in range(0, some_w.size):
    some_loss[i] = L(some_w[i], m, y)

plt.figure(2)
plt.plot(some_w, some_loss, '-r', linewidth=2)
plt.xlabel('w')
plt.ylabel('loss')

# Gradient descent
w_tilde = np.random.rand(1)
minVal = 1e-9
alpha = 0.01
while (L(w_tilde, m, y) > minVal):
    w_tilde = w_tilde - alpha * (1/m * np.sum((h(w_tilde) - y) * x))

print('optimal w_tilde: ' + str(w_tilde))


y_tilde = h(w_tilde)

plt.figure(1)
plt.plot(x, y_tilde, '-b', linewidth =2, label='learned model')
plt.legend()