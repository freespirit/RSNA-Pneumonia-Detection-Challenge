import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os

sns.set(style="white", palette="muted", color_codes=True)

# %%
# Read the data
df_train_labels_orig = pd.read_csv("data/stage_1_train_labels.csv")

# %%
# Explore the data
df_train_labels_orig.head()

positive_samples = df_train_labels_orig[df_train_labels_orig['Target'] == 1]
negative_samples = df_train_labels_orig[df_train_labels_orig['Target'] == 0]
print(f"Positive: {positive_samples.count()}; Negative: {negative_samples.count()}")

# %%
counts = np.array([positive_samples.shape[0], negative_samples.shape[0]])
labels = np.array(["Positive", "Negative"])
plot = sns.barplot(x = labels, y = counts, saturation=.5)
plot.set_title("Training Samples")
fig = plot.get_figure()
fig.savefig("screenshots/samples.png")

# %%
points = positive_samples[['x', 'y']]

f, axes = plt.subplots(2, 2, figsize=(10, 10), sharey = 'row')
axes[0, 0].set_title('X distribution')
axes[0, 1].set_title('Y distribution')
axes[1, 0].set_title('Bounding box distribution')
axes[1, 1].set_title('Bounding box distribution')

sns.distplot(points['x'], kde=False, color="b", ax=axes[0,0])
sns.distplot(points['y'], kde=False, color="b", ax=axes[0,1])
sns.scatterplot(x = "x", y = "y", data = points, ax=axes[1,0])
sns.kdeplot(points['x'], points['y'], shade=True, ax=axes[1,1])
f.savefig("screenshots/box_coordinates.png")


# %%
widths = positive_samples['width']
heights = positive_samples['height']

fig, axes = plt.subplots(1, 2, figsize = (10, 5), sharey=True)
axes[0].set_title('width distribution')
axes[1].set_title('height distribution')
sns.distplot(widths, kde=False, ax=axes[0])
sns.distplot(heights, kde=False, ax=axes[1])
fig.savefig("screenshots/box_sides.png")

fig, axes = plt.subplots(1, 2, figsize = (10, 5))
axes[0].set_title('area distribution')
axes[1].set_title('ratio distribution')
sns.distplot(widths*heights, kde=False, ax=axes[0])
sns.distplot(heights/widths, kde=False, ax=axes[1])
fig.savefig("screenshots/box_shapes.png")
