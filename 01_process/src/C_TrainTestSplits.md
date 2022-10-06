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
from pylab import cm

np.random.seed(123)
```

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
date_format = '%Y-%m-%d'

vars_to_keep = ['ShortWave', 'LongWave', 'AirTemp', 'RelHum', 'WindSpeed', 'Rain',
                'Snow', 'ice', 'temp_0_x', 'MaxDepth', 'LakeArea']
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

### Randomly sample lakes until we get 20% of the data

```python
def partition_by_lakes(dataset, possible_lakes, desired_prop):
    
    # arguments
    #     `dataset` is the dataset we're partitioning.
    #        I assume it has a "DOW" column
    #     `possible_lakes` are the DOWs that we sample.
    #        As the sampling continues, we remove values.
    #     `desired_prop` is the proportion of the data set
    #        that we want this partition to cover.
    # returns
    #     `partition_lakes` are the random DOWs representing
    #        the desired proportion of the data set
    #     `possible_lakes` are the reduced DOWs that we can
    #        further sample for additional partitions.
    
    partition_lakes = []
    cumulative_sum = 0 

    while cumulative_sum < 0.2:
        # add a random new lake to the partition
        rand_DOW = np.random.choice(possible_DOWs)
        partition_lakes.append(rand_DOW)

        # subset the data set to those sampled partitions
        # and determine what % we are at
        cumulative_subset = dataset[dataset.DOW.isin(partition_lakes)]
        cumulative_sum = np.sum(cumulative_subset['prop'])

        # sample without replacement
        possible_DOWs.remove(rand_DOW)

    # State what literal proportion we achieved and
    # how many lakes that represents
    print(cumulative_sum, len(partition_lakes))
    
    return(partition_lakes, possible_lakes)
```

```python
possible_DOWs = list(DOW_proportions['DOW'].copy())

valid_lakes, possible_DOWs = partition_by_lakes(DOW_proportions,
                                                possible_DOWs,
                                                0.2)
test_lakes, possible_DOWs = partition_by_lakes(DOW_proportions,
                                               possible_DOWs,
                                               0.2)
```

```python
DOW_proportions['train'] = DOW_proportions.DOW.isin(possible_DOWs)
DOW_proportions['valid'] = DOW_proportions.DOW.isin(valid_lakes)
DOW_proportions['test'] = DOW_proportions.DOW.isin(test_lakes)
```

# Very basic mapping of the partitions

```python
lat_lon_ref_df = lat_lon_ref_df.rename(columns = {'dow num':'DOW'})
lat_lon_ref_df = lat_lon_ref_df.merge(DOW_proportions, on='DOW')

seq_DOWs = pd.DataFrame({'DOW':DOW})

mapping_df = seq_DOWs.merge(lat_lon_ref_df, on='DOW', how = 'left')
mapping_df.head()
```

```python
fig, ax = plt.subplots(1, 3, figsize = (15, 5))
cmap = cm.get_cmap('viridis', 2)

count = 0
for col in ['train', 'valid', 'test']:
    ax[count].set_title(col + ' lake locations (yellow)\nN sequences = ' + str(np.sum(mapping_df[col])))
    ax[count].scatter(mapping_df['long'], mapping_df['lat'], cmap = cmap,
                      c = mapping_df[col], marker = '^')
    count += 1
    
print('WARNING: Some lakes may not be displayed due to missing values in the lat/lon\n' + 
      '         data set, but they are in-fact present in the model-ready partitions.')
```

# Actually split the model ready sequences

```python
train_indices = []
valid_indices = []
test_indices = []

count = 0
for dow in DOW:
    if dow in possible_DOWs:
        train_indices.append(count)
    elif dow in valid_lakes:
        valid_indices.append(count)
    elif dow in test_lakes:
        test_indices.append(count)
    else:
        print('WARNING', count)
        
    count += 1

len(train_indices), len(valid_indices), len(test_indices)
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
save_partition_data(train_indices, train_data_fpath)
save_partition_data(valid_indices, valid_data_fpath)
save_partition_data(test_indices, test_data_fpath)
```

```python

```
