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
train_out_dir = '../../02_train/out/'

# data, primarily for the ice flags
train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_predictions_fpath = train_out_dir + 'massive_lstm_train_preds_1_.npy'
valid_predictions_fpath = train_out_dir + 'massive_lstm_valid_preds_1_.npy'

eval_metrics_fpath = '../out/massive_lstm_eval_metrics_1_.npz'

mapping_reference = "../../01_process/in/MN_ice/raw_data_from_DNR/lake_ice_id_spreadsheet.xlsx"
```

# Import data

```python
# Import both development partitions
train_data = np.load(train_data_fpath, allow_pickle = True)
valid_data = np.load(valid_data_fpath, allow_pickle = True)

# Extract everything from training partition
train_x = train_data['x']
train_y = train_data['y']
train_dates = train_data['dates']
train_DOW = train_data['DOW']
train_variables = train_data['features']

# Extract everything from validation partition
valid_x = valid_data['x']
valid_y = valid_data['y']
valid_dates = valid_data['dates']
valid_DOW = valid_data['DOW']
valid_variables = valid_data['features']
```

```python
train_predictions = np.load(train_predictions_fpath)
valid_predictions = np.load(valid_predictions_fpath)
```

```python
# Programmatically identify process-based ice flag data
ice_var_idx = int(np.argwhere(train_variables == 'ice'))
depth_var_idx = int(np.argwhere(train_variables == 'MaxDepth'))
area_var_idx = int(np.argwhere(train_variables == 'LakeArea'))
assert valid_variables[ice_var_idx] == 'ice'
assert valid_variables[depth_var_idx] == 'MaxDepth'
assert valid_variables[area_var_idx] == 'LakeArea'
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
    rand_i = np.random.choice(valid_predictions.shape[0])

    plt.figure(figsize = (15, 2))
    plt.axhline(0.5, label = '50% prob', color = 'black')
    plt.plot(valid_y[rand_i, :], label = 'obs', linewidth = 3)
    plt.plot(valid_predictions[rand_i, :, 0], label = 'pred prob')
    plt.plot(valid_x[rand_i, :, ice_var_idx], label = 'ice_flag', linestyle = '--')
    plt.title('DOW: ' + str(valid_DOW[rand_i]) + ', Start date: ' + valid_dates[rand_i][0])
    plt.legend();
```

# Spatial distribution

```python
# Read in the MN lake metadata, focusing on coordiantes and lake size
lat_lon_ref_df = pd.read_excel(mapping_reference,
                               usecols=['dow num', 'lat', 'long', 'acres'])
lat_lon_ref_df = lat_lon_ref_df.rename(columns = {'dow num':'DOW'})

# Merge that information with the validation set DOWs
seq_DOWs = pd.DataFrame({'DOW':valid_DOW})
mapping_df = seq_DOWs.merge(lat_lon_ref_df, on='DOW', how = 'left')
```

```python
# Assign all the errors to that newly merged dataframe
# (both are sorted by validation set DOW)
mapping_df['flag_error_ice_on'] = eval_metrics['flag_error_ice_on']
mapping_df['flag_error_ice_off'] = eval_metrics['flag_error_ice_off']
mapping_df['flag_error_dur']  = eval_metrics['flag_error_dur'] 
mapping_df['pred_error_ice_on']  = eval_metrics['pred_error_ice_on'] 
mapping_df['pred_error_ice_off'] = eval_metrics['pred_error_ice_off'] 
mapping_df['pred_error_dur'] = eval_metrics['pred_error_dur']
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
areas = valid_x[:, 0, area_var_idx]

plot_and_print_resid_corr(areas, 'Lake Area\nlog-transformed')
```

```python
depths = valid_x[:, 0, depth_var_idx]

plot_and_print_resid_corr(depths, 'Lake Maximum Depth\nlog-transformed')
```

## List of significant residual correlations

##### Latitude

* The process-based model's and the massive lstm's residuals are signficantly correlated with latitude on all 3 date-based predictions
    * Negatively correlated for ice on, positively correlated for ice off and ice duration
  
##### Longitude
  
* The process-based model's residuals are significantly and positively correlated with longitude for ice on prediction

##### Lake area

* The process-based model's residuals are significantly correlated with lake area on all 3 date-based predictions
    * Negatively correlated for ice on, positively correlated for ice off and ice duration
    
##### Lake depth
    
* The process-based model's residuals are significantly correlated with lake depth on ice on and ice duration prediction
    * Negatively correlated for ice on, positively correlated for ice duration
* The massive lstm's residuals are significantly and positively correlated for ice on prediction

### In total

* The process-based model's residuals are significantly correlated with static lake descriptions in 9/12 tested scenarios, most notably latitude and lake area for all 3 date-based predictions.
* The massive lstm's residuals are significantly correlated with static lake descriptions in 4/12 tested scenarios, with 3 of those scenarios involving latitude.

```python

```
