import matplotlib.pyplot as plt
import numpy as np

#Data
x = np.array([2023, 2024, 2025, 2026, 2028])
y1 = np.array([15, 25, 30, 20, 30])
y2 = np.array([17, 27, 10, 40, 67])
y3 = np.array([27, 17, 30, 20, 2])

#Title customization
plt.title("Class Size",
          fontsize=25,
          family="Arial",
          fontweight="bold",
          color="grey"
          )

#Axis customization
plt.xlabel("Year",
           fontsize=20,
          family="Arial",
          fontweight="bold",
          color="lightblue")

plt.ylabel("Students",
           fontsize=20,
          family="Arial",
          fontweight="bold",
          color="lightblue")

#Grid function
plt.grid(axis="y",
         linewidth=2,
         color="lightgreen",
         linestyle="dashed")

#General reusable styling
line_style = dict(marker=".",
         markersize=30, # can be represented as ms
         markerfacecolor="#ebeb34", # can be represented as mfc
         markeredgecolor="#0bf334",
         linestyle="solid",
         linewidth=4,
         color="#0000cf")

#Tick customization
plt.tick_params(axis="both", colors="lightblue")

#Plot definition
plt.plot(x, y1, **line_style)
plt.plot(x, y2, **line_style)
plt.plot(x, y3, **line_style)

#Only show exact data ticks
# plt.xticks(x)

#Open window
plt.show()

