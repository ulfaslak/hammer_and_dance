import matplotlib.pylab as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import ConvexHull
from scipy import stats
from utils import *


# --------------------- #
# Load and process data #
# --------------------- #

data = pd.read_csv(f'data_proc/data.csv', index_col=0)
data.index = pd.to_datetime(data.index)


# -------------------------------- #
# Plot symptoms and cases together #
# -------------------------------- #

# split into input and output
data_x = data.filter(regex="cli.+|ili.+|mask.+")
data_y = data.filter(regex="cases.+")

# smoothen cases as 7-day online moving average
data_x = data_x.rolling(7, min_periods=1).mean()
data_y = data_y.rolling(7, min_periods=1).mean()

for c in tqdm(sorted(set(["_".join(c.split("_")[1:]) for c in data.columns]))):
    
    data_c = pd.concat([data_x[f'cli_{c}'], data_y[f'cases_{c}']], axis=1).dropna()
    lag, maxr = highest_correlation_lag(
        data_c.values[:, 0], data_c.values[:, 1], bounds=[-40, 40])

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1 = (data_x[f'cli_{c}']*100).plot(
        ax=ax1, color="red", label="% reporting COVID-like symptoms")
    (data_c[f'cli_{c}'].shift(-lag)*100).plot(
        ax=ax1, color="red", lw=0.3, label=f"lag={lag}")
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    ax2 = (data_y[f'cases_{c}']*100000).plot(ax=ax2,
                                               color="blue", label="% new cases")
    ax1.set_ylabel('Percent', fontsize=12)
    ax2.set_ylabel('Per 100.000', fontsize=12)
    plt.title(c.replace("_", " "))
    fig.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{c}_covid_symptoms_vs_cases.png")
    plt.close()

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1 = (data_x[f'mask_{c}']*100).plot(
        ax=ax1, color="red", label="% reporting mask use", ls="dotted")
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    ax2 = (data_y[f'cases_{c}']*100000).plot(
        ax=ax2, color="blue", label="% new cases")
    ax1.set_ylabel('Percent', fontsize=12)
    ax2.set_ylabel('Per 100.000', fontsize=12)
    plt.title(c.replace("_", " "))
    fig.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{c}_masks_vs_cases.png")
    plt.close()


# --------------------------- #
# Plot symptoms vs cases rate #
# --------------------------- # 

# Min filter data
min_date = pd.to_datetime('2020-08-01')
data_x = data.loc[data.index > min_date].filter(regex="cli.+")
data_y = data.loc[data.index > min_date].filter(regex="cases.+")
countries = sorted(set(["_".join(c.split("_")[1:]) for c in data.columns]))

# Colormap
cmap = {
    'Africa': "#AC92EB",
    'Americas': "#4FC1E8",
    'Asia': "#A0D568",
    'Europe': "#FFCE54",
    'Oceania': "#ED5564"}

# Aggregate poins for each world region
region_points = defaultdict(list)
for country in countries:
    try:
        region = CountryInfo(country.replace("_", " ")).region()
    except KeyError:
        continue
    data_x_country = data_x.filter(regex=country).values * 100
    data_y_country = data_y.filter(regex=country).values * 100000
    data_xy_country = np.hstack([data_x_country, data_y_country])
    region_points[region].append(data_xy_country)

# Plot
fig, ax = plt.subplots(figsize=(6, 4))
for region, data_region in region_points.items():
    data_region = np.vstack(data_region)
    data_region = data_region[~np.any(np.isnan(data_region), axis=1)]
    points = peel_to_confidence(data_region, confidence=0.7, centroid_func=np.median, scaling_func=(np.log, np.exp))
    hull = ConvexHull(points)
    ax.scatter(data_region[:, 0], data_region[:, 1], c=cmap[region], s=3)
    ax.fill(points[hull.vertices, 0], points[hull.vertices, 1],
            facecolor='none', edgecolor=cmap[region], lw=2, label=region)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel('Symptoms per 100 capita')
plt.ylabel('Cases per 100.000 capita')
plt.show()


# ------------------------------------- #
# Plot ratio between symptoms and cases #
# ------------------------------------- #

# Min filter data
min_date = pd.to_datetime('2020-04-01')
data_x = data.loc[data.index > min_date].filter(regex="cli.+")
data_y = data.loc[data.index > min_date].filter(regex="cases.+")
countries = sorted(set(["_".join(c.split("_")[1:]) for c in data.columns]))

# Colormap
cmap = {
    'Africa': "#AC92EB",
    'Americas': "#4FC1E8",
    'Asia': "#A0D568",
    'Europe': "#FFCE54",
    'Oceania': "#ED5564"}

# Aggregate poins for each world region
country_points = defaultdict(list)
region_points = defaultdict(list)
for country in tqdm(countries):
    try:
        region = CountryInfo(country.replace("_", " ")).region()
    except KeyError:
        continue
    data_x_country = data_x.filter(regex=country).values
    data_y_country = data_y.filter(regex=country).values
    ratios = (data_y_country / data_x_country).reshape(-1)
    ratios = [v for v in ratios if not np.isinf(v)]
    country_points[country].extend(ratios)
    region_points[region].extend(ratios)

# Plot cases per reported symptoms
fig, ax = plt.subplots(figsize=(6, 4))
sorted_regions_points_items = sorted(region_points.items(), key=lambda kv: np.mean(kv[1]))
for i, (region, data_region) in enumerate(sorted_regions_points_items):
    y, x = np.histogram(data_region, density=False, bins=np.logspace(-6, 0, 31))
    x, y = unzip([(x_, y_) for x_, y_ in zip(x[1:], y) if y_ > 0])
    plt.plot(x, y, color=cmap[region], label=region)
plt.xscale("log")
plt.yscale("log")
plt.legend()
plt.xlabel("cases / symptom reports")
plt.ylabel("count frequency")
plt.show()

# Plot for countries
fig, ax = plt.subplots(figsize=(6, 4))
sorted_countries_points_items = sorted(
    country_points.items(), key=lambda kv: np.mean(kv[1]))
for i, (country, data_country) in enumerate(sorted_countries_points_items):
    y, x = np.histogram(data_country, density=False, bins=np.logspace(-6, 0, 31))
    x, y = unzip([(x_, y_) for x_, y_ in zip(x[1:], y) if y_ > 0])
    plt.plot(
        x, y,
        color=cmap[CountryInfo(country.replace("_", " ")).region()],
        label=country)
plt.xscale("log")
plt.yscale("log")
# plt.legend()
plt.xlabel("cases / symptom reports")
plt.ylabel("count frequency")
plt.show()


# ---------------------------------- #
# Plot optimal time lag distribution #
# ---------------------------------- #

data_interpolated = data.interpolate()

countries = sorted(set(["_".join(c.split("_")[1:]) for c in data.columns]))

# Compute maxr lag and aggregate on regions
lag_vals = []
maxr_vals = []
for country in tqdm(countries):
    try:
        region = get_region(country)
    except KeyError:
        continue
    data_country = data[[f'cli_{country}', f'cases_{country}']].dropna()
    lag, maxr = highest_correlation_lag(
        data_country[f'cli_{country}'],
        data_country[f'cases_{country}'],
        bounds=[-40, 40]
    )
    lag_vals.append(lag)
    maxr_vals.append(maxr)

# Plot maxr lag distributions 
fig, ax = plt.subplots(figsize=(6, 4))
plt.hist(lag_vals, density=False, bins=np.linspace(-40, 40, 21))
plt.legend()
plt.xlabel("optimal lag")
plt.ylabel("count frequency")
plt.show()
