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
```

# What happens in this notebook?

1. Additional static variables are added - lake area and lake depth
2. Certain variables are removed (due to their missingness or their expected availability outside the study area)
3. Data is split into train, validation, and test partitions
  * The test partition consists of lakes and years that are not represented in the validation and train partitions
    * Notably, the test partition consists of the most recent years (2010-2017)
  * The validation and train partitions share lakes, but contain mutually exclusive years
    * Validation = 2005-2009
    * Train = 1980-2004


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

# compatible with earlier efforts, leaves enough to train and eval with
# a 'soft' test set can be larger, consisting of train/eval lakes during test years
# and test lakes during train/eval years
test_set_prop = 0.08

test_start_year = 2010
valid_start_year = 2005

# num times to attempt to find a test set that maximizes training data
try_n_test_partitions = 10000
```

### Outputs

```python
train_data_fpath = out_dir + 'train_data.npz'
valid_data_fpath = out_dir + 'valid_data.npz'
test_data_fpath = out_dir + 'test_data.npz'
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

# Sample different lakes into the test set many times

Always achieve a desired test set proportion, then track valid/train size. After lots of iterations, select the sample of test set lakes that maximizes training data.

Years are set/fixed based on a preference for evaluating future years. 8 years (2010-2017, inclusive) was seen as a minimum good test set.

```python
def partition_by_lakes(dataset, possible_lakes, desired_prop, total_N, rng):
    
    # arguments
    #     `dataset` is the dataset we're partitioning.
    #        I assume it has a "DOW" column
    #     `possible_lakes` are the DOWs that we sample.
    #        As the sampling continues, we remove values.
    #     `desired_prop` is the proportion of the data set
    #        that we want this partition to cover.
    #     `total_N` is the number of available sequences across
    #        all lakes and years
    #     `rng` is a np.random.RandomState object (opposed to
    #        np.random.seed setting)
    # returns
    #     `partition_lakes` are the random DOWs representing
    #        the desired minimum proportion of the data set
    
    partition_lakes = []
    cumulative_sum = 0 

    while cumulative_sum < desired_prop:
        # add a random new lake to the partition
        rand_DOW = rng.choice(possible_lakes)
        partition_lakes.append(rand_DOW)

        # subset the data set to those sampled partitions
        # and determine what % we are at
        cumulative_subset = dataset[dataset.DOW.isin(partition_lakes)]
        cumulative_sum = cumulative_subset.shape[0] / total_N

        # sample without replacement
        possible_lakes.remove(rand_DOW)
    
    return(partition_lakes)
```

```python
dev_period = df[df['year'] < test_start_year]
test_period = df[df['year'] >= test_start_year]
```

```python
RNG = np.random.RandomState(random_seed)
```

```python
%%time

test_lake_candidates = []
partition_sizes = np.zeros([try_n_test_partitions, 3])

for count in range(try_n_test_partitions):
    # sample random lakes that are observed in the test years for possible exclusion
    # from train/valid sets
    test_lakes = partition_by_lakes(test_period,
                                    list(test_period['DOW'].copy()),
                                    test_set_prop,
                                    df.shape[0],
                                    RNG)

    # Make the test set consist exclusive lakes (on top of exclusive years)
    test_df = test_period[test_period['DOW'].isin(test_lakes)]

    # Train/valid will be earlier period omitting test lakes
    dev_df = dev_period[dev_period['DOW'].isin(test_lakes) == False]

    # Separate train/valid by year
    valid_df = dev_df[dev_df['year'] >= valid_start_year]
    train_df = dev_df[dev_df['year'] < valid_start_year]

    # Keep track of each random attempt...
    # ...partition sizes (relative to whole)
    partition_sizes[count] = (train_df.shape[0] / df.shape[0],
                              valid_df.shape[0] / df.shape[0],
                              test_df.shape[0] / df.shape[0])
    # ...test set lakes omitted
    test_lake_candidates.append(test_lakes)
```

```python
# what maximizes the training set with the desired test set prop?
found_iteration = np.argmax(partition_sizes[:, 0])
test_lakes = test_lake_candidates[found_iteration]

# 50/16.5/8 split
partition_sizes[found_iteration, :]
```

```python
# test years and test lakes
test_df = test_period[test_period['DOW'].isin(test_lakes)]

# Train/valid will be earlier period omitting test lakes
dev_df = dev_period[dev_period['DOW'].isin(test_lakes) == False]

# Separate train/valid by year
valid_df = dev_df[dev_df['year'] >= valid_start_year]
train_df = dev_df[dev_df['year'] < valid_start_year]
```

```python
# plot the partitions
fig, ax = plt.subplots(1, 2, figsize = (12, 6))

ax[0].scatter(train_df['long'], train_df['lat'], s = 10, label = 'train')
ax[0].scatter(valid_df['long'], valid_df['lat'], s = 10, label = 'valid')
ax[0].scatter(test_df['long'], test_df['lat'], s = 10, label = 'test')
ax[0].legend();

# transparency to demostrate no overlap
plt.hist(train_df['year'], bins = np.arange(1980, 2020), label = 'train', alpha = 0.75)
plt.hist(valid_df['year'], bins = np.arange(1980, 2020), label = 'eval', alpha = 0.75)
plt.hist(test_df['year'], bins = np.arange(1980, 2020), label = 'test', alpha = 0.75)
plt.legend();
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

<br><br><br><br>

# Lake overlap checking

```python
# Percent of valid lakes appearing in test set
len([dow for dow in valid_df['DOW'].unique() if dow in test_df['DOW'].unique()]) / valid_df['DOW'].unique().shape[0]
```

```python
# Percent of valid lakes appearing in train set
#   Not 100% due to timing of available observations
len([dow for dow in valid_df['DOW'].unique() if dow in train_df['DOW'].unique()]) / valid_df['DOW'].unique().shape[0]
```

```python
# Percent of train lakes appearing in test set
len([dow for dow in test_df['DOW'].unique() if dow in train_df['DOW'].unique()]) / test_df['DOW'].unique().shape[0]
```
