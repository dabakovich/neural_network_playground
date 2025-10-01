import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Створюємо фігуру та 3D осі
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Точки
points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 1]])
ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
           c=['red', 'red', 'red', 'blue'], s=100, marker='o', label='Точки')


# Створюємо площину z = x + y - 1.5
x = np.linspace(-0.5, 1.5, 50)
y = np.linspace(-0.5, 1.5, 50)
X, Y = np.meshgrid(x, y)
Z = -X -Y + 1.5

# Малюємо площину
ax.plot_surface(X, Y, Z, alpha=0.5, cmap='viridis', label='z = -x - y + 1.5')

# Налаштування
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('4 точки та площина z = -x - y + 1.5')
ax.legend()

plt.show()