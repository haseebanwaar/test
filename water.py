
print("chnagded")
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0, 100)
y = x * 0.4 + 10

plt.yticks(range(0,600,50))
plt.plot(x, y, 'x-')
plt.show()