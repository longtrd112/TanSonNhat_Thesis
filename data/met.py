import pandas as pd
import matplotlib.pyplot as plt

# x = ['Wind speed', 'Wind direction', 'Visibility', 'Cloud coverage lv1', 'Cloud coverage lv2', 'Cloud coverage lv3', 'Total']
# y = [1, 16910 - 12678, 16910 - 16907, 208, 6728, 15753, 16910]
# fig, ax = plt.subplots()
# ax.bar(x, y)
# for bar in ax.patches:
#     ax.annotate(text=bar.get_height(),
#                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
#                 ha='center',
#                 va='center',
#                 size=10,
#                 xytext=(0, 8),
#                 textcoords='offset points')
# plt.ylabel("Number of unavailable data points")
# # plt.title("Unusable flights removal")
# plt.show()

# df = pd.read_csv('sortedData/2021-03-01/VJ139.csv')
# x = df.timestamp.to_list()
# t = []
# for a in x:
#     t.append(a-x[0])
# y = df.vertical_rate.to_list()
# b = df.altitude.to_list()
#
#
# # create figure and axis objects with subplots()
# fig, ax = plt.subplots()
# # make a plot
# lns1 = ax.plot(t, y, color="red", label="Vertical Rate")
# # set x-axis label
# ax.set_xlabel("Time")
# # set y-axis label
# ax.set_ylabel("Vertical Rate", color="red")
#
# # twin object for two different y-axis on the sample plot
# ax2 = ax.twinx()
# # make a plot with different y-axis using second axis object
# lns2 = ax2.plot(t, b, color="blue", label="Altitude")
# ax2.set_ylabel("Altitude", color="blue")
#
# lns = lns1+lns2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
# plt.title('2021-03-01: VJ139')
# plt.show()

df = pd.read_csv('VVTS_metar.csv')
skyc = []
for i in range(len(df)):
    if df.skyc1.iloc[i] == "None":
        df.skyc1.iloc[i] = None
df.info()
df.to_csv("VVTS_metar.csv", index=False)
