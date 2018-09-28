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

# %%
counts = np.array([positive_samples.shape[0], negative_samples.shape[0]])
labels = np.array(["Positive", "Negative"])
plot = sns.barplot(x = labels, y = counts, saturation=.5)
plot.set_title("Training Samples")

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


#%%
widths = positive_samples['width']
heights = positive_samples['height']

f, axes = plt.subplots(1, 2, figsize = (10, 5), sharey=True)
axes[0].set_title('width distribution')
axes[1].set_title('height distribution')
sns.distplot(widths, kde=False, ax=axes[0])
sns.distplot(heights, kde=False, ax=axes[1])

f, axes = plt.subplots(1, 1, figsize = (5, 5))
axes.set_title('area distribution')
sns.distplot(widths*heights, kde=False)
