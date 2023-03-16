import os
import pandas as pd
from featureExtraction.utils.plotMAP import plotMAP, plt

copy_data_directory = "../data/ExtractCopy"

error = pd.read_csv('../data/flightError.csv')
a = error.error.value_counts().to_dict()
a['Less than 50 data points'] = 74265 - 1365 - 61688
a['Total'] = 74265

courses = list(a.keys())
values = list(a.values())
fig, ax = plt.subplots()
ax.bar(courses, values)
for bar in ax.patches:
    ax.annotate(text=bar.get_height(),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center',
                va='center',
                size=12,
                xytext=(0, 8),
                textcoords='offset points')
plt.ylabel("Number of flights")
plt.title("Unusable flights removal")
plt.show()
