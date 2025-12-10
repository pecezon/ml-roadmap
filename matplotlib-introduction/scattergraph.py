import matplotlib.pyplot as plt
import numpy as np

# Data
x1 = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 9, 10])
x2 = np.array([0, 1, 2, 2, 3, 4, 5, 4, 10, 7, 9, 10])
y1 = np.array([50, 60, 65, 63, 68, 70, 75, 78, 82, 85, 87, 90])
y2 = np.array([55, 58, 60, 62, 66, 69, 73, 77, 80, 83, 86, 88])

# Scatter Plot
plt.scatter(x1, y1,
            color="#ff5733",
            alpha=0.5,
            s=100,
            label="Group 1",
            )

plt.scatter(x2, y2,
            color="#33c1ff",
            alpha=0.5,
            s=100,
            label="Group 2",)

plt.title("Study Hours vs Test Scores",)
plt.xlabel("Hours Studied")
plt.ylabel("Test Scores")

#Show legend
plt.legend()

plt.show()