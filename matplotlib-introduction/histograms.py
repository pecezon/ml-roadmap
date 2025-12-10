import matplotlib.pyplot as plt
import numpy as np

# Data

#loc is the mean, scale is the standard deviation, and size is the number of samples
scores = np.random.normal(80, 10, 100)
scores = np.clip(scores, 0, 100)

plt.hist(scores,
         bins=20,
         color="#ff5733",
         edgecolor="black",)

plt.title("Test Score Distribution",
          fontsize=25,
          family="Arial",
          fontweight="bold",
          color="grey"
          )
plt.xlabel("Scores",
           fontsize=20,
           family="Arial",
           fontweight="bold",
           color="lightblue")
plt.ylabel("Number of Students",
           fontsize=20,
           family="Arial",
           fontweight="bold",
           color="lightblue")

plt.show()