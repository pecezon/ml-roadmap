import matplotlib.pyplot as plt
import numpy as np

# Data
categories = np.array(["Grains", "Vegetables", "Fruits", "Dairy", "Meat", "Sweets", "Beverages"])
values = np.array([50, 80, 70, 60, 90, 40, 30])

# Vertical Bar Chart
# plt.bar(categories, values,
#         color="#4caf50")

# Horizontal Bar Chart
plt.barh(categories, values,
        color="#4caf50")

# Title and Labels
plt.title("Food Consumption")
plt.xlabel("Food Categories")
plt.ylabel("Consumption Level")

plt.show()