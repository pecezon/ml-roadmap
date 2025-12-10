import matplotlib.pyplot as plt
import numpy as np

# Data
categories = np.array(["Freshmen", "Sophomores", "Juniors", "Seniors"])
values = np.array([120, 90, 70, 50])

# Pie Chart
colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

plt.pie(values, labels=categories,
        autopct="%1.1f%%",
        colors=colors,
        explode=[0, 0, 0, 0.1],
        shadow=True,
        startangle=140,)

plt.title("Class Distribution",
          fontsize=20,
            family="Arial",
            fontweight="bold",
            color="grey"
            )

plt.show()