from collections import Counter
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import seaborn as sns


def plot_categoricals(data):
    ncols = len(data.columns)
    fig = plt.figure(figsize=(5 * 5, 5 * (ncols // 5 + 1)))
    for i, col in enumerate(data.columns):
        cnt = Counter(data[col])
        keys = list(cnt.keys())
        vals = list(cnt.values())
        plt.subplot(ncols // 5 + 1, 5, i + 1)
        plt.bar(range(len(keys)), vals, align="center")
        plt.xticks(range(len(keys)), keys)
        plt.xlabel(col, fontsize=18)
        plt.ylabel("frequency", fontsize=18)
    fig.tight_layout()
    plt.show()


def create_heatmap(data):
    plt.figure(figsize=(11, 9))
    plt.title('Correlation between feature columns')
    corr = data.apply(lambda x: x.factorize()[0]).corr()
    sns.heatmap(corr)
    plt.show()


def create_scatterplot(data):
    sns.scatterplot(data=data, x='plus_minus_home Rolling Avg',
                    y='record_home', hue='Home Win/Loss', alpha=1)
    plt.show()


cleaned_data = pd.read_csv('../data/final_data.csv')

# plot_categoricals(cleaned_data)
# create_heatmap(cleaned_data)
# create_scatterplot(cleaned_data)
