---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import numpy as np
import datetime 
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
```

# What happens in this notebook?

1. Additional static variables are added - lake area and lake depth
2. Certain variables are removed (due to their missingness or their expected availability outside the study area)
3. Data is split into train, validation, and test partitions
  * The train partition consists of lakes and years that are not represented in the validation and test partitions
    * Notablly, all training data consists of years that predate validation and test data (2001 and earlier)
  * The validation and test partitions share lakes, but contain mutually exclusive years
    * Validation = 2002-2006
    * Test = 2007-2015


# Configuration


### Inputs

```python
out_dir = "../out/"
model_ready_data_fpath = out_dir + 'model_ready_sequences.npz'
matching_df_fpath = out_dir + "matching_sources.csv"

mapping_reference = "../in/MN_ice/raw_data_from_DNR/lake_ice_id_spreadsheet.xlsx"
```

### Values

```python
random_seed = 123

date_format = '%Y-%m-%d'

vars_to_keep = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain',
                'Snow', 'ice', 'temp_0_x', 'MaxDepth', 'LakeArea']

# need a decently long evaluation period from random search
min_years_partition = 10

# favorite output from the seed-set random search
found_iteration = 262
```

### Outputs

```python
train_data_fpath = out_dir + 'train_data_updated.npz'
valid_data_fpath = out_dir + 'valid_data_updated.npz'
test_data_fpath = out_dir + 'test_data_updated.npz'
```

# Import

```python
model_ready_data = np.load(model_ready_data_fpath, allow_pickle = True)

lat_lon_ref_df = pd.read_excel(mapping_reference,
                               usecols=['dow num', 'lat', 'long'])
```

```python
x = model_ready_data['x']
y = model_ready_data['y']
dates =  model_ready_data['dates']
DOW =  model_ready_data['DOW']
variables =  model_ready_data['features']

print('Number of sequences before this work... ', str(x.shape[0]))
```

# Add in area and depth from TOHA metadata

```python
matching_df = pd.read_csv(matching_df_fpath)
matching_df
```

```python
assert np.sum(np.isnan(matching_df['area'])) == np.sum(np.isnan(matching_df['depth'])) == 0
```

```python
depth_vals = np.ones([x.shape[0], x.shape[1]])
area_vals = np.ones([x.shape[0], x.shape[1], 1])

count = 0
for dow in DOW:
    depth_val = matching_df[matching_df['DOW'] == dow]['depth'].item()
    area_val = matching_df[matching_df['DOW'] == dow]['area'].item()
    
    depth_vals[count] = depth_val*depth_vals[count]
    area_vals[count] = area_val*area_vals[count]
    
    count += 1
```

```python
plt.hist(depth_vals.flatten())
plt.pause(0.0001)
plt.hist(np.log(depth_vals.flatten()));
```

```python
plt.hist(area_vals.flatten())
plt.pause(0.0001)
plt.hist(np.log(area_vals.flatten()));
```

```python
# overwrite the existing max depth (associated with deep water temp estimnates)
x[:, :, np.argwhere(variables == 'MaxDepth').item()] = np.log(depth_vals)

# add in lake area
x = np.concatenate([x, np.log(area_vals)], axis = 2)
variables = np.concatenate([variables, ['LakeArea']])
```

# Demonstrate existing data missingness

```python
# We dont really need the 366th day after July 1 on leap years
#   because it will almost surely be ice free.
# Additionally, the 366th day is nan for non leap years.
temporary_x = x[:, :365, :].copy()

# Determine how many sequences have a NaN present for each variable
for i in range(len(variables)):
    nan_free_seq_count = 0
    for j in range(temporary_x.shape[0]):
        if np.sum(np.isnan(temporary_x[j, :, i])) == 0:
            nan_free_seq_count += 1
    print(i, variables[i], '\t\t', np.round(nan_free_seq_count / temporary_x.shape[0], 2))
```

**Working decision:** dont use PGDL estimates for water temperature because they are missing too often.


# Omit variables or rows that provide missingness


### Omit variables

```python
vars_to_keep_ids = np.isin(variables, vars_to_keep)
```

```python
non_omitted_variables = variables[vars_to_keep_ids]
print(non_omitted_variables)

improved_x = temporary_x[:, :, vars_to_keep_ids]

del(temporary_x)
```

### Omit rows

```python
# Determine where data is missing and what percent are missing
nan_free_seq_count = 0
nan_free_indices = []
for j in range(improved_x.shape[0]):
    if np.sum(np.isnan(improved_x[j])) == 0:
        nan_free_seq_count += 1
        nan_free_indices.append(j)
nan_free_indices = np.asarray(nan_free_indices)

perc_missing = 1  - (nan_free_seq_count / improved_x.shape[0])
perc_missing
```

```python
improved_x = improved_x[nan_free_indices]
# Also get rid of that extra day on leap years here
y = y[nan_free_indices, :365]
dates = dates[nan_free_indices, :365]
DOW = DOW[nan_free_indices].astype(int)
variables = non_omitted_variables
```

```python
print('Number of sequences remaining... ', str(improved_x.shape[0]))
```

# Identify partitions by lake identifier (DOW)

Train, validation, and test sets will be mutually exclusive sets of different lakes. Data from these lake sets should approximately span the entire temporal range

```python
df = pd.DataFrame({"DOW":DOW.astype(int),
                   "dates":dates[:, 0].astype(np.datetime64)})
df = df.dropna()
df['dates'] = df['dates'].astype(np.datetime64)

lat_lon_ref_df = lat_lon_ref_df.rename(columns = {'dow num':'DOW'})
df = df.merge(lat_lon_ref_df, how = 'left', on = 'DOW')

df['year'] = [date.year for date in df['dates']]
```

### Figure out what % of the data each lake represents

```python
DOW_value_counts = df['DOW'].value_counts()
DOW_proportions = DOW_value_counts / len(df)
DOW_proportions = pd.DataFrame(DOW_proportions)
DOW_proportions = DOW_proportions.reset_index()
DOW_proportions = DOW_proportions.rename(columns = {'DOW':'prop',
                                                    'index':'DOW'})
DOW_proportions
```

```python
df = df.merge(DOW_proportions, how = 'left', on = 'DOW')
```

# Utility functions

```python
# get min and max from values that potentially have missing values
def get_min_and_max(vals):
    
    vals_min = np.nanmin(vals)
    vals_max = np.nanmax(vals)
    
    return vals_min, vals_max
```

```python
# generate random intervals within a min-max range
def generate_random_bounds(vals):
    
    vals_min, vals_max = get_min_and_max(vals)
    
    val_rand_start = np.random.uniform(vals_min, vals_max)
    val_rand_end = np.random.uniform(val_rand_start, vals_max)
    
    return val_rand_start, val_rand_end
```

```python
# simple function to subset dataframe
def subset_by_year(data, years):

    years_min, years_max = years
    
    sub_data = data[(data['year'] > years_min)*
                    (data['year'] < years_max)]
    
    return sub_data
```

```python
# simple function to get data outside the subset above
def get_remaining_data(data, years):
    
    years_min, years_max = years
    
    sub_indices = ((data['year'] > years_min)*
                   (data['year'] < years_max))
    
    remaining_indices = np.asarray([not i for i in sub_indices])
    
    remaining_data = data[remaining_indices]
    
    return remaining_data
```

# Perform a random search of different year-bound partitions

```python
np.random.seed(random_seed)
```

```python
%%time
n_tries = 500 # the nature of the algorithm does not result in many unique solutions

parameters = np.zeros([n_tries, 2]) # year min+maxes
partition_sizes = np.zeros([n_tries, 2]) # size of each partition
dow_tracking = []

for k in range(n_tries):
    possible_DOWs = list(DOW_proportions['DOW'].copy())
    
    # generate random year bounds
    year_bounds_A = generate_random_bounds(df['year'])
    # ensure the bounds contain some minimum amount of years
    while np.diff(year_bounds_A) <= min_years_partition:
        year_bounds_A = generate_random_bounds(df['year'])

    # subset to year
    partition_A = subset_by_year(df, year_bounds_A)
    partition_B = get_remaining_data(df, year_bounds_A)

    # identify the lakes in each partition
    # and calculate what percent of the data set 
    # they comprise
    DOW_value_counts_A = partition_A['DOW'].value_counts()
    DOW_proportions_A = DOW_value_counts_A / len(df)
    DOW_proportions_A = pd.DataFrame(DOW_proportions_A)
    DOW_proportions_A = DOW_proportions_A.reset_index()
    DOW_proportions_A = DOW_proportions_A.rename(columns = {'DOW':'prop',
                                                        'index':'DOW'})

    DOW_value_counts_B = partition_B['DOW'].value_counts()
    DOW_proportions_B = DOW_value_counts_B / len(df)
    DOW_proportions_B = pd.DataFrame(DOW_proportions_B)
    DOW_proportions_B = DOW_proportions_B.reset_index()
    DOW_proportions_B = DOW_proportions_B.rename(columns = {'DOW':'prop',
                                                        'index':'DOW'})
    
    # identify only the lakes in one partition
    dow_in_A_only = [dow for dow in DOW_proportions_A['DOW'] if dow not in DOW_proportions_B['DOW'].values]
    dow_tracking.append(dow_in_A_only)
    
    # Track year bounds
    parameters[k] = year_bounds_A
    # Calculate how much data remains when partition A (bounded by year_bounds_A)
    # uses only the lakes unique to its years and excludes those lakes
    # from partition B (the training set)
    partition_sizes[k] = [np.sum(DOW_proportions_B[DOW_proportions_B['DOW'].isin(dow_in_A_only) == False]['prop']),
                          np.sum(DOW_proportions_A[DOW_proportions_A['DOW'].isin(dow_in_A_only)]['prop'])]
```

```python
# Interactive plot to select a random search iteration
viz_df = pd.DataFrame(partition_sizes)
viz_df = viz_df.reset_index()
viz_df = viz_df.rename(columns = {0:'train', 1:'eval'})

px.scatter(viz_df, x='train', y='eval', hover_name = 'index')
```

```python
# Manually inspected index from interactive plot
# seed is set, so this wont change
year_bounds_A = parameters[found_iteration]
cur_DOW_A = dow_tracking[found_iteration]
```

```python
# separate the evaluation and training data
eval_df = df[df['DOW'].isin(cur_DOW_A)]
eval_df = subset_by_year(eval_df, year_bounds_A)
train_df = get_remaining_data(df, year_bounds_A)
```

```python
# plot the evaluation and training data
fig, ax = plt.subplots(1, 2, figsize = (12, 6))

ax[0].scatter(train_df['long'], train_df['lat'], s = 10, label = 'train')
ax[0].scatter(eval_df['long'], eval_df['lat'], s = 10, label = 'eval')
ax[0].legend();

plt.hist(train_df['year'], bins = np.arange(1980, 2020), label = 'train')
plt.hist(eval_df['year'], bins = np.arange(1980, 2020), label = 'eval')
plt.legend();
```

```python
# go ahead and make train past-only since it's very close to that
train_df = train_df[train_df['year'] < 2015]

# give validation a few future years
valid_df = eval_df[eval_df['year'] <= 2006]

# give test the latest years
test_df = eval_df[eval_df['year'] > 2006]
```

```python
# View latest temporal split
plt.hist(train_df['year'], bins = np.arange(1980, 2020), label = 'train')
plt.hist(valid_df['year'], bins = np.arange(1980, 2020), label = 'validation')
plt.hist(test_df['year'], bins = np.arange(1980, 2020), label = 'test')
plt.legend();

# Print proportions (41-9-7; lost about 1/3 of data because exclusive lakes and times)
print('Train percent of whole:', np.round(train_df.shape[0] / df.shape[0], 3))
print('Valid percent of whole:', np.round(valid_df.shape[0] / df.shape[0], 3))
print('Test percent of whole:', np.round(test_df.shape[0] / df.shape[0], 3))
```

```python
def save_partition_data(indices, fpath):
    # subset all the objects
    part_x = improved_x[indices]
    part_y = y[indices]
    part_dates = dates[indices]
    part_DOW = DOW[indices]
    
    part_data = {'x':part_x,
                 'y':part_y,
                 'dates':part_dates,
                 'DOW':part_DOW,
                 'features':non_omitted_variables}
    
    np.savez_compressed(fpath, **part_data)
```

```python
save_partition_data(train_df.index, train_data_fpath)
save_partition_data(valid_df.index, valid_data_fpath)
save_partition_data(test_df.index, test_data_fpath)
```

```python

```
