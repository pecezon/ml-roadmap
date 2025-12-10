import matplotlib.pyplot as plt
import numpy as np

# Figure = Entire window or page
# Axes = Single plotting area

x1 = np.array([0, 1, 1, 2, 3, 4, 5, 6, 7, 7, 9, 10])
y1 = np.array([50, 60, 65, 63, 68, 70, 75, 78, 82, 85, 87, 90])

figure, axes = plt.subplots(2, 2)

axes[0, 0].plot(x1, y1,
                color="#ff5733",
                )
axes[0, 0].set_title("Study Hours vs Test Scores")

axes[0, 1].plot(x1, x1**2, color="blue")
axes[0, 1].set_title("X vs X Squared")

axes[1, 0].plot(x1, np.sqrt(x1), color="green")
axes[1, 0].set_title("X vs Square Root of X")

axes[1, 1].plot(x1, np.log1p(x1), color="purple")
axes[1, 1].set_title("X vs Log(X + 1)")

plt.tight_layout()
plt.show()