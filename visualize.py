import matplotlib.pylab as plt
import pandas as pd

# Load and process data
# ---------------------

data = pd.read_csv(f'data_proc/data.csv', index_col=0)

# split into input and output
data_in = data.filter(regex="symptoms.+")
data_out = data.filter(regex="cases.+")

# smoothen cases as 7-day online moving average
data_in = data_in.rolling(7, min_periods=1).mean()
data_out = data_out.rolling(7, min_periods=1).mean()

for c in sorted(set(["_".join(c.split("_")[1:]) for c in data.columns])):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1 = data_in[f'symptoms_{c}'].plot(ax=ax1, color="red", label="% with COVID-like symptoms")
    ax2 = ax1.twinx()
    ax2.spines['right'].set_position(('axes', 1.0))
    ax2 = data_out[f'cases_{c}'].plot(ax=ax2, color="blue", label="# new daily cases")
    ax1.set_ylabel('Percent', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    plt.title(c.replace("_", " "))
    fig.legend()
    plt.tight_layout()
    plt.savefig(f"plots/{c}_symptoms_vs_cases.png")
