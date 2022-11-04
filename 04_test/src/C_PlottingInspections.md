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
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr
```

# Configuration


### Inputs

```python
process_out_dir = '../../01_process/out/'
test_out_dir = '../../04_test/out/'

# data, primarily for the ice flags
soft_test_data_fpath = process_out_dir + 'soft_test_data.npz'
test_data_fpath = process_out_dir + 'test_data.npz'

# associated predictions
soft_test_preds_fpath = test_out_dir + 'massive_lstm_soft_test_preds_1_.npy'
test_preds_fpath = test_out_dir + 'massive_lstm_test_preds_1_.npy'

eval_metrics_fpath = '../out/massive_lstm_test_metrics_1_.npz'

mapping_reference = "../../01_process/in/MN_ice/raw_data_from_DNR/lake_ice_id_spreadsheet.xlsx"
```

# Import data

```python
# Import both testing partitions
soft_test_data = np.load(soft_test_data_fpath, allow_pickle = True)
test_data = np.load(test_data_fpath, allow_pickle = True)

# Extract everything from soft testing partition
soft_test_x = soft_test_data['x']
soft_test_y = soft_test_data['y']
soft_test_dates = soft_test_data['dates']
soft_test_DOW = soft_test_data['DOW']
soft_test_variables = soft_test_data['features']

# Extract everything from testing partition
test_x = test_data['x']
test_y = test_data['y']
test_dates = test_data['dates']
test_DOW = test_data['DOW']
test_variables = test_data['features']

# Import associated predictions
soft_test_preds = np.load(soft_test_preds_fpath)
test_preds = np.load(test_preds_fpath)
```

```python
# Separate the two types of soft testing data
train_years_test_lakes_ids = np.argwhere(soft_test_data['dates'][:, 0] < test_data['dates'].min()).flatten()
test_years_train_lakes_ids = np.argwhere(soft_test_data['dates'][:, 0] >= test_data['dates'].min()).flatten()

train_years_test_lakes_preds = soft_test_preds[train_years_test_lakes_ids]
train_years_test_lakes_x = soft_test_x[train_years_test_lakes_ids]
train_years_test_lakes_y = soft_test_y[train_years_test_lakes_ids]
train_years_test_lakes_dates = soft_test_dates[train_years_test_lakes_ids]

test_years_train_lakes_preds = soft_test_preds[test_years_train_lakes_ids]
test_years_train_lakes_x = soft_test_x[test_years_train_lakes_ids]
test_years_train_lakes_y = soft_test_y[test_years_train_lakes_ids]
test_years_train_lakes_dates = soft_test_dates[test_years_train_lakes_ids]
```

```python
# Programmatically identify process-based ice flag data
ice_var_idx = int(np.argwhere(test_variables == 'ice'))
depth_var_idx = int(np.argwhere(test_variables == 'MaxDepth'))
area_var_idx = int(np.argwhere(test_variables == 'LakeArea'))
assert soft_test_variables[ice_var_idx] == 'ice'
assert soft_test_variables[depth_var_idx] == 'MaxDepth'
assert soft_test_variables[area_var_idx] == 'LakeArea'
```

```python
eval_metrics = np.load(eval_metrics_fpath, allow_pickle = True)
```

# Plot some simple time series

```python
# Seed to view the same ones each time
np.random.seed(123)

# 10 random time series plots
# Includes:
#   Observation
#   Prediction
#   PB ice flag
#   50% probability line
#   Date and DOW of time series
for i in range(20):
    rand_i = np.random.choice(test_preds.shape[0])

    plt.figure(figsize = (15, 2))
    plt.axhline(0.5, label = '50% prob', color = 'black')
    plt.plot(test_y[rand_i, :], label = 'obs', linewidth = 3)
    plt.plot(test_preds[rand_i, :, 0], label = 'pred prob')
    plt.plot(test_x[rand_i, :, ice_var_idx], label = 'ice_flag', linestyle = '--')
    plt.title('DOW: ' + str(test_DOW[rand_i]) + ', Start date: ' + test_dates[rand_i][0])
    plt.xlim(100, 325)
    plt.legend();
```

# Spatial distribution

```python
# Read in the MN lake metadata, focusing on coordiantes and lake size
lat_lon_ref_df = pd.read_excel(mapping_reference,
                               usecols=['dow num', 'lat', 'long', 'acres'])
lat_lon_ref_df = lat_lon_ref_df.rename(columns = {'dow num':'DOW'})

# Merge that information with the validation set DOWs
seq_DOWs = pd.DataFrame({'DOW':test_DOW})
mapping_df = seq_DOWs.merge(lat_lon_ref_df, on='DOW', how = 'left')
```

```python
# Assign all the errors to that newly merged dataframe
# (both are sorted by validation set DOW)
mapping_df['flag_error_ice_on'] = eval_metrics['test_flag_error_ice_on']
mapping_df['flag_error_ice_off'] = eval_metrics['test_flag_error_ice_off']
mapping_df['flag_error_dur']  = eval_metrics['test_flag_error_dur'] 
mapping_df['pred_error_ice_on']  = eval_metrics['test_pred_error_ice_on'] 
mapping_df['pred_error_ice_off'] = eval_metrics['test_pred_error_ice_off'] 
mapping_df['pred_error_dur'] = eval_metrics['test_pred_error_dur']
```

```python
def map_error(error_col_suffix, jitter):
    # Craft the two column names
    process_based_col = 'flag_' + error_col_suffix
    new_model_col = 'pred_' + error_col_suffix
    
    # Set the colors to map to the same values
    cmap_extreme = np.max(np.abs([mapping_df[new_model_col].min(),
                                  mapping_df[new_model_col].max()]))
    
    # Configure a 2-plot arangement
    fig, ax = plt.subplots(1, 3, figsize = (13, 6), gridspec_kw={'width_ratios': [6, 6, 0.5]})
    
    # Create the spatial jitter values (same for both subplots)
    lat_jitter = np.random.normal(loc = 0, scale = jitter, size = mapping_df.shape[0])
    lon_jitter = np.random.normal(loc = 0, scale = jitter, size = mapping_df.shape[0])
    
    # Plot ice flag errors with spatial jitter
    # Size and color are mapped to error
    ax[0].set_title("Process-based Ice Flags")
    ax[0].scatter(mapping_df['long'] + lon_jitter,
                  mapping_df['lat'] + lat_jitter,
                  s = np.abs(mapping_df[process_based_col]*3),
                  marker = 'o',
                  cmap = 'BrBG',
                  vmin = -1*cmap_extreme,
                  vmax = cmap_extreme,
                  c = mapping_df[process_based_col])
    
    # Plot new prediction errors with spatial jitter
    # Size and color are mapped to error
    ax[1].set_title("New Model Predictions")
    im = ax[1].scatter(mapping_df['long'] + lon_jitter,
                  mapping_df['lat'] + lat_jitter,
                  s = np.abs(mapping_df[new_model_col]*3),
                  marker = 'o',
                  cmap = 'BrBG',
                  vmin = -1*cmap_extreme,
                  vmax = cmap_extreme,
                  c = mapping_df[new_model_col])
    
    # Color legend
    fig.colorbar(im, label = 'Days', cax = ax[2]);
```

```python
map_error('error_ice_on', 0.2)
```

```python
map_error('error_ice_off', 0.2)
```

```python
map_error('error_dur', 0.2)
```

# Do residuals significantly vary with certain lake characteristics?

Some additional evaluation for how residuals change with certain lake characteristics: lat, long, depth, and area. These residual-inspecting plots include a nonlinear correlation and p-value

```python
def plot_and_print_resid_corr(values, label):
    
    # globally accessed mapping_df, assuming its column names (residuals), and associated axis labels
    
    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    
    fig.suptitle('Correlation of Residuals with Static Lake Descriptions')

    ax[0].scatter(values, mapping_df['flag_error_ice_on'], alpha = 0.5, label = 'Process-based')
    ax[1].scatter(values, mapping_df['flag_error_ice_off'], alpha = 0.5)
    ax[2].scatter(values, mapping_df['flag_error_dur'], alpha = 0.5)
    ax[0].scatter(values, mapping_df['pred_error_ice_on'], alpha = 0.5, label = 'LSTM')
    ax[1].scatter(values, mapping_df['pred_error_ice_off'], alpha = 0.5)
    ax[2].scatter(values, mapping_df['pred_error_dur'], alpha = 0.5)
    ax[0].legend()

    ax[0].set_ylabel('Residual')
    ax[0].set_title('Ice on')
    ax[1].set_title('Ice off')
    ax[1].set_xlabel(label, fontsize = 12)
    ax[2].set_title('Ice duration')
    
    plt.show()

    print('\nProcess-based residual correlations')
    print('Ice on:\t\t', spearmanr(values, mapping_df['flag_error_ice_on']))
    print('Ice off:\t', spearmanr(values, mapping_df['flag_error_ice_off']))
    print('Ice duration:\t', spearmanr(values, mapping_df['flag_error_dur']))
    print('\nLSTM-based residual correlations')
    print('Ice on:\t\t', spearmanr(values, mapping_df['pred_error_ice_on']))
    print('Ice off:\t', spearmanr(values, mapping_df['pred_error_ice_off']))
    print('Ice duration:\t', spearmanr(values, mapping_df['pred_error_dur']), '\n')
```

```python
plot_and_print_resid_corr(mapping_df['lat'], 'Latitude')
```

```python
plot_and_print_resid_corr(mapping_df['long'], 'Longitude')
```

```python
areas = test_x[:, 0, area_var_idx]

plot_and_print_resid_corr(areas, 'Lake Area\nlog-transformed')
```

```python
depths = test_x[:, 0, depth_var_idx]

plot_and_print_resid_corr(depths, 'Lake Maximum Depth\nlog-transformed')
```

## List of significant residual correlations

##### Latitude

* The process-based model's residuals are signficantly correlated with latitude on ice on and ice duration prediction. 
    * Negatively correlated for ice on, positively correlated for ice duration
* The massive lstm's residuals are significantly and positively correlated with latitude on ice off and ice duration prediction.
  
##### Longitude
  
* The process-based model's residuals are significantly and positively correlated with longitude for ice on prediction

##### Lake area

* No significant correlations.
    
##### Lake depth
    
* The process-based model's residuals are significantly and negatively correlated with lake depth on ice on prediction

### In total

* The process-based model's residuals are significantly correlated with static lake descriptions in 4/12 tested scenarios, half of these scenarios involve latitude.
* The massive lstm's residuals are significantly correlated with static lake descriptions in 2/12 tested scenarios, both of these scenarios involve latitude.


<br><br><br>

### Paper figure

```python
fig, ax = plt.subplots(5, 1, figsize = (12, 12))

count = 0
# different years, different lakes, different behavior
# 264 - pretty clean behavior, lot of improvement
# 148 - improvement but not perfect, displays two PB fall freezes
# 331 - nearly perfect
# 339 - moderate improvement ice on, weakened ice off
# 99 - difficult but improved ice on, equal ice off

for i in [264, 148, 331, 339, 99]:

    ax[count].axhline(0.5, label = '50% prob', linestyle = '--', color = 'black')
    ax[count].plot(test_y[i, :], label = 'Observation', linewidth = 3)
    ax[count].plot(test_preds[i, :, 0], label = 'LSTM %')
    ax[count].plot(test_x[i, :, ice_var_idx], label = 'Process-based')
    ax[count].set_ylabel('Probably (or identifier)\nof ice', fontsize = 10)
    ax[count].set_title('DOW: ' + str(test_DOW[i]) + ', Start date: ' + test_dates[i][0])
    ax[count].set_xlim(100, 325)
    if count == 0:
        ax[count].legend()
    if count == 4:
        ax[count].set_xlabel('Days after July 1', fontsize = 12)
    count += 1

fig.tight_layout()

plt.savefig('../../handpicked_test_timeseries.PNG', dpi = 300, bbox_inches = 'tight')
```

```python

```
