import matplotlib.pyplot as plt
import numpy as np

# fig = plt.figure(figsize=(10, 5))  # create figure handle
# ax = fig.add_subplot(231)  # create axis handle
# ax.set_xlim()
# ax.set_xticks()

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# create data
x = np.linspace(-np.pi, np.pi, 100)
y = np.sin(x)

# line plot
ax1.plot(x, y, linewidth=2.0, color='b')

# add noise
y += 0.3*np.random.randn(y.shape[0])

# scatter plot
ax2.scatter(x, y, s=20, color='r')

# IGNORE THESE
plt.tight_layout()
plt.show()
