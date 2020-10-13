import requests as rq
import json
import pandas as pd
from tqdm import tqdm
import numpy as np


def unzip(zipped_list):
    return list(map(list, zip(*zipped_list)))

# Earliest date to keep in dataset
min_date = pd.to_datetime('2020-03-01')

# Get case data for the word
cases = pd.read_excel(
    'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-10-12.xlsx')
cases = cases.loc[cases['dateRep'] > min_date]

# Stringency index
stringency = pd.read_csv(
    'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv')
stringency['Date'] = pd.to_datetime(stringency['Date'], format="%Y%m%d")
stringency = stringency.loc[stringency['Date'] > min_date]

# Loop over countries to match case data with symptom data
# avg_sample_size = []
df_li, countries = [], []
for country in tqdm(cases['countriesAndTerritories'].unique()):

    # CASE DATA
    cases_country = cases.loc[cases['countriesAndTerritories'] == country]
    date_range_l = str(
        cases_country['dateRep'].iloc[-1]).split()[0].replace("-", "")
    date_range_u = str(cases_country['dateRep'].iloc[0]).split()[
        0].replace("-", "")

    # SURVEY DATA
    symptoms_country_covid = rq.get(
        "https://covidmap.umd.edu/api/resources?" +
        "indicator=covid&" +
        "type=daily&" +
        f"country={country}&" +
        f"daterange={date_range_l}-{date_range_u}",
        headers={'User-agent': "ulfaslak"}
    ).json()
    symptoms_country_mask = rq.get(
        "https://covidmap.umd.edu/api/resources?" +
        "indicator=mask&" +
        "type=daily&" +
        f"country={country}&" +
        f"daterange={date_range_l}-{date_range_u}",
        headers={'User-agent': "ulfaslak"}
    ).json()

    # skip at threshold
    if len(symptoms_country_covid['data']) < 160:
        continue

    # process symptoms and mask data
    time_cli, percent_cli = unzip([
        (pd.to_datetime(d['survey_date']), d['percent_cli'])
        for d in symptoms_country_covid['data']
    ])
    time_mask, percent_mask = unzip([
        (pd.to_datetime(d['survey_date']), d['percent_mc'])
        for d in symptoms_country_mask['data']
    ])

    # process stringency
    stringency_country = stringency.loc[stringency['CountryName'] == country.replace(
        "_", " ").title()]

    # Skip if no stringency index for country
    if stringency_country.shape[0] == 0:
        print(f"no str ind for {country}")
        continue

    # append
    df_c = pd.concat([
        pd.DataFrame(percent_cli, index=time_cli, columns=[f'cli_{country}']),
        pd.DataFrame(percent_mask, index=time_mask, columns=[f'mask_{country}']),
        pd.DataFrame(stringency_country['StringencyIndex'].values, index=stringency_country['Date'], columns=[f'stringency_{country}']).groupby('Date').max(),
        pd.DataFrame(cases_country['cases'].values / cases_country['popData2019'].values, index=cases_country['dateRep'], columns=[f'cases_{country}']),
    ], axis=1, join='outer')
    df_li.append(df_c)
    countries.append(country)

# concat
df = pd.concat(df_li, axis=1, join='outer')

# remove rows with all nans
df = df.loc[~df.isna().all(1)]

df.to_csv(f'data_proc/data.csv')



# # ----------------------------- #
# # Add stringency and norm cases #
# # ----------------------------- #

# # Load proc data
# df = pd.read_csv(f'data_proc/data.csv', index_col=0)

# # Cases
# cases = pd.read_excel(
#     'https://www.ecdc.europa.eu/sites/default/files/documents/COVID-19-geographic-disbtribution-worldwide-2020-10-07.xlsx')

# # Stringency index
# stringency = pd.read_csv(
#     'https://raw.githubusercontent.com/OxCGRT/covid-policy-tracker/master/data/OxCGRT_latest.csv')
# stringency['Date'] = pd.to_datetime(stringency['Date'], format="%Y%m%d")

# df_li, countries = [], []
# for country in sorted(set(["_".join(c.split("_")[1:]) for c in df.columns])):
    
#     # Stringency and cases
#     stringency_country = stringency.loc[stringency['CountryName'] == country.replace("_", " ").title()]
#     cases_country = cases.loc[cases['countriesAndTerritories'] == country]

#     if stringency_country.shape[0] == 0:
#         print(f"no str ind for {country}")
#         continue

#     df_li.append(
#         pd.DataFrame(df[f'cli_{country}']))
#     df_li.append(
#         pd.DataFrame(df[f'mask_{country}']))
#     df_li.append(
#         pd.DataFrame(stringency_country['StringencyIndex'].values, index=stringency_country['Date'].values, columns=[f'stringency_{country}']))
#     df_li.append(
#         pd.DataFrame(df[f'cases_{country}'] / cases_country['popData2019'].values[0]))


# for i in tqdm(range(len(df_li)), total=len(df_li)):
#     df_new = pd.concat(df_li[:i+1], axis=1, join="outer")

# df_li[i]
