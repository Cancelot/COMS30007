import numpy as np
import matplotlib.pyplot as plt

M = np.sort(np.random.randn(256, 256))

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
im1 = ax1.imshow(M)

ax2 = fig.add_subplot(122)
im2 = ax2.imshow(20 * M)
ax2.set_yticks([])

plt.colorbar(im1, ax=ax1)
plt.colorbar(im2, ax=ax2)

plt.show()
