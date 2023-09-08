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
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt

import os
```

# Configuration

"Data release" refers to https://www.sciencebase.gov/catalog/item/5e5c1b4fe4b01d50924f27e1


### Inputs

```python
ice_obs_fpath = "../in/MN_ice/ice_duration_summarized.csv"

prior_data_release_dir = "../in/prior_data_release/"
data_release_metadata_fpath = prior_data_release_dir + "lake_metadata.csv"
mapping_reference = "../in/MN_ice/raw_data_from_DNR/lake_ice_id_spreadsheet.xlsx"
```

### Values

```python
use_lat = True
```

### Outputs

```python
out_dir = "../out/"
out_driver_dir = out_dir + "merged_drivers/by_DOW/"
missing_pgdl_fpath = out_dir + "DOW_missing_PGDL_estimates.npy"

if use_lat:
    matching_df_fpath = out_dir.replace("out/", "out_WithLat/") + "matching_sources.csv"
else:
    matching_df_fpath = out_dir + "matching_sources.csv"
```

# Import base data sets

```python
ice_dur = pd.read_csv(ice_obs_fpath)
data_release_metadata = pd.read_csv(data_release_metadata_fpath)
```

```python
# how many unique DOWs exist in the ice targets data set?
ice_dur['DOW'].unique().shape
```

# Create a new data set for easily referencing between the observations and drivers

The main usefulness here is that the ice observations are provided by Minnesota DOW identifiers while the TOHA data release drivers are primarily sorted by NHD identifiers. Sometimes one NHD identifier corresponds to multiple DOW identifiers, so this code unravels that duplication (i.e., individual rows rather than rows with list values) - being more repetitive but more easily accessible from the DOW point of view.


### Check that the same DOW is not matched to multiple NHDHRs

```python
# Get all the DOWs by row/NHDHR
DOW_nums_ls = []
for DOW in data_release_metadata['mndow_id']:
    DOW_nums = DOW.replace("|", "").split('mndow_')[1:]
    # unnesting the multiple lake aggregating and adding rows for them (and their files)
    DOW_nums_ls.append(DOW_nums)
```

```python
# Preview
DOW_nums_ls[:5]
```

```python
# Check if the same DOW can be matched to other NHDHRs

check_count = 0
# for every DOW tuple (possibly singular)
for DOW_tuple in DOW_nums_ls:
    # for every item in that tuple
    for DOW in DOW_tuple:
        compare_count = 0
        for other_DOW_tuple in DOW_nums_ls:
            # ignore when they're considering the same tuple
            if compare_count != check_count:
                # make sure the DOW is not in the other tuples
                assert DOW not in other_DOW_tuple
            compare_count += 1
    check_count += 1
```

### Check NHDHR column for consistent naming

```python
for val in data_release_metadata['site_id']:
    assert val[:6] == 'nhdhr_' # check consistent format
```

### Determine matching

```python
DOW_nums_ls = []
group_ls = []
meteo_file_ls = []
nhdhr_ls = []
depth_ls = []
area_ls = []

for DOW in data_release_metadata['mndow_id']:
    DOW_nums = DOW.replace("|", "").split('mndow_')[1:]
    # unnesting the multiple lake aggregating and adding rows for them (and their files)
    for ID in DOW_nums:
        DOW_nums_ls.append(int(ID))
        group_ls.append(data_release_metadata[data_release_metadata['mndow_id'] == DOW]['group_id'].item())
        meteo_file_ls.append(data_release_metadata[data_release_metadata['mndow_id'] == DOW]['meteo_filename'].item())
        nhdhr_ls.append(data_release_metadata[data_release_metadata['mndow_id'] == DOW]['site_id'].item()[6:])
        depth_ls.append(data_release_metadata[data_release_metadata['mndow_id'] == DOW]['depth'].item())
        area_ls.append(data_release_metadata[data_release_metadata['mndow_id'] == DOW]['area'].item())
        
matching_df = pd.DataFrame({'DOW':DOW_nums_ls,
                            'group':group_ls,
                            'meteo_file':meteo_file_ls,
                            'nhdhr':nhdhr_ls,
                            'depth':depth_ls,
                            'area':area_ls})
matching_df
```

### Append latitude

```python
# Read in the MN lake metadata, focusing on coordiantes and lake size
lat_lon_ref_df = pd.read_excel(mapping_reference,
                               usecols=['dow num', 'lat'])
lat_lon_ref_df = lat_lon_ref_df.rename(columns = {'dow num':'DOW'})

# Merge that information with the DOWs
matching_df = matching_df.merge(lat_lon_ref_df, on='DOW', how = 'left')
matching_df
```

```python
# save it for later use
matching_df.to_csv(matching_df_fpath)
```

# For each DOW identifier, aggregrate and store all the data release drivers

```python
# inner merge omits any lakes that do not have
# corresponding values in the both data sources
ice_dur = ice_dur.merge(matching_df,
                        how = 'inner',
                        on = 'DOW')
```

```python
# how many unique DOWs and NHDHRs made it through the merge?
ice_dur['DOW'].unique().shape, ice_dur['nhdhr'].unique().shape
```

```python
missing_pgdl = []

count = 0
for DOW in ice_dur['DOW'].unique():
    
    # subset the df for value grabbing
    cur_df = ice_dur[ice_dur['DOW'] == DOW].reset_index(drop=True)
    
    group = cur_df['group'].unique().item()
    meteo_file = cur_df['meteo_file'].unique().item()
    nhdhr = cur_df['nhdhr'].unique().item()
    
    # get the meteo inputs
    meteo = pd.read_csv(prior_data_release_dir +
                        'inputs_' +
                        group +
                        '/' +
                        meteo_file)

    # get the clarity inputs
    clarity = pd.read_csv(prior_data_release_dir +
                          'clarity_' +
                          group +
                          '/gam_nhdhr_' +
                          nhdhr + 
                          '_clarity.csv') 

    # get the PB ice flags
    ice_flags = pd.read_csv(prior_data_release_dir +
                            'ice_flags_' +
                            group +
                            '/pb0_nhdhr_' +
                            nhdhr + 
                            '_ice_flags.csv') 

    # get the PB irradiance
    irradiance = pd.read_csv(prior_data_release_dir +
                             'irradiance_' +
                             group +
                             '/pb0_nhdhr_' +
                             nhdhr + 
                             '_irradiance.csv')   

    # get the PB water temps
    PB_water_temps = pd.read_csv(prior_data_release_dir +
                                 'pb0_predictions_' +
                                 group +
                                 '/pb0_nhdhr_' +
                                 nhdhr + 
                                 '_temperatures.csv')

    # get the PGDL water temps WHEN THEY ARE AVAILABLE
    try:
        PGDL_water_temps = pd.read_csv(prior_data_release_dir +
                                       'pgdl_predictions_' +
                                       group + 
                                       '/tmp/pgdl_nhdhr_' +
                                       nhdhr + 
                                       '_temperatures.csv')
    # otherwise proceed without them
    except FileNotFoundError as e:
        missing_pgdl.append(DOW)
        PGDL_water_temps = None

    # merge them all
    meteo = meteo.rename(columns = {'time':'date'})
    inputs = meteo.merge(clarity, on = 'date', how = 'left')
    inputs = inputs.merge(ice_flags, on = 'date', how = 'left')
    inputs = inputs.merge(irradiance, on = 'date', how = 'left')
    inputs = inputs.merge(PB_water_temps, on = 'date', how = 'left')
    try:
        inputs = inputs.merge(PGDL_water_temps, on = 'date', how = 'left')
    except TypeError as e:
        pass

    inputs.to_csv(out_driver_dir + "DOW_" + str(DOW) + "_all_vars.csv")
```

```python
# How many lakes (that made it through the merge) lacked PGDL estimates?
len(missing_pgdl)
```

```python
np.save(missing_pgdl_fpath, missing_pgdl)
```

Originally, I merged these by nhdhr, but I ultimately decided on DOW as the identifier because DOW is what we have labels for, so it seems more important to be faithful to the labels than the drivers (which are by nhdhr)


<br><br><br><br><br><br><br><br><br>

# Some Quality Checks

Mostly concerned with PGDL estimates

### Quality checking where we have missing data

Identify all the unique lakes and count how many ice observations each unique lake has

```python
count_ls = []
for DOW in ice_dur['DOW'].unique():
    count_ls.append(ice_dur[ice_dur['DOW'] == DOW]['DOW'].value_counts().item())
```

```python
new_df = pd.DataFrame({'DOW':ice_dur['DOW'].unique(),
                       'count':count_ls})
```

Now, match that information with which unique lakes are lacking pgdl driver data

```python
indices = []
for i in range(new_df.shape[0]):
    indices.append(new_df['DOW'][i] in missing_pgdl)
```

```python
new_df[indices].shape
```

```python
np.sum(new_df['count'][indices]), np.sum(new_df['count'][indices]) / ice_dur.shape[0]
```

The unique lakes that lack any PGDL estimates amount to 975 observations, which is 19% of the data set. This pays no attention to the timing of PGDL estimates or observations.


<br><br><br>

### Quality checking when we have missing data

Same procedure as above

BUT, first remove years that don't have any driver data anyways

```python
relevant_years = ice_dur[(ice_dur['min_ice_on_date'] > '1980-01-01') * (ice_dur['min_ice_on_date'] <= '2018-07-01')]
```

```python
count_ls = []
for DOW in relevant_years['DOW'].unique():
    count_ls.append(relevant_years[relevant_years['DOW'] == DOW]['DOW'].value_counts().item())
```

```python
new_df = pd.DataFrame({'DOW':relevant_years['DOW'].unique(),
                       'count':count_ls})
```

```python
np.sum(new_df['count']), (ice_dur.shape[0] - np.sum(new_df['count'])) / ice_dur.shape[0]
```

Years that are outside the temporal bounds of the driver data represent approximately 15% of the data set alone.

Beyond that...

```python
indices = []
for i in range(new_df.shape[0]):
    indices.append(new_df['DOW'][i] in missing_pgdl)
```

```python
# what?? it should be 15%
# how can 15% of sequences have non-NA pgdl but 20% of them are missing files
np.sum(new_df['count'][indices]), np.sum(new_df['count'][indices]) / np.sum(new_df['count'])
```

An additional ~20% lack PGDL estimates due to their lakes not having PGDL estimates for any time period.

So ~35% of observations do not have PGDL estimates.

```python

```
