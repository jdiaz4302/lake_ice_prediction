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
from collections import OrderedDict
import pandas as pd 
from sklearn.metrics import r2_score
```

# Configuration


### Inputs

```python
process_out_dir = '01_process/out/'
train_out_dir = '02_train/out/'

# data, primarily for the ice flags
train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'
```

```python
extended_dir = '/caldera/projects/usgs/water/iidd/datasci/lake-temp/lake_ice_prediction/'

train_data_fpath = extended_dir + train_data_fpath
valid_data_fpath = extended_dir + valid_data_fpath
```

### Values

```python
# when deriving the max ice on date, one detail is that
# we must omit the late ice on dates that occur during a
# (predicted) rethaw. This value is a temporal index
# representing the maximum day after July 1 that can
# be a considered a max ice on date.
# A value of 215 equates to February 1
ice_on_cutoff = 215

# Use to contruct 1-day bins on the histogram from (-num, num)
n_days_hist_cutoff = 75


# Information needed to construct relevant filepaths
num_init = 5
pred_fpath_start = ['avg', 'large', 'massive']
model_names = ['lstm', 'transformer']


# Possibly fragile: sizes needed to store all preds together
seq_len = 365
train_n = 2178
valid_n = 722
n_epochs = 10000
n_compare = num_init*len(pred_fpath_start)*len(model_names)

# color guide
color_dict = {'avg lstm':'#1b9e77',
              'large lstm':'#7570b3',
              'massive lstm':'#66a61e',
              'avg transformer':'#d95f02',
              'large transformer':'#e7298a',
              'process-based':'#a6761d',
              'massive transformer':'#e6ab02',}

# remove process-based or not
remove_PB = True
# this corresponds to some known indices (dependent on data assembly steps)
# of bad model performance to remove because they complicate analysis
# 11 never predicts freeze
# 13 predicts freeze but never predicts thaw
bad_remove_PB_wout_Lat_indices = [11, 13]

use_lat = True
if use_lat:
    train_out_dir = train_out_dir.replace("out", "out_WithLat")
    train_data_fpath = train_data_fpath.replace("out", "out_WithLat")
    valid_data_fpath = valid_data_fpath.replace("out", "out_WithLat")
```

### Outputs

```python
# Fragile: prepare to change if new results; this is a little qualitative atm so not automated
if use_lat:
    if remove_PB:
        second_best_eval_metrics_fpath = '../out_WithLat/massive_lstm_eval_metrics_4_NoProcessBasedInput_.npz'
        best_eval_metrics_fpath = '../out_WithLat/massive_lstm_eval_metrics_3_NoProcessBasedInput_.npz'
        eval_table_fpath = '../out_WithLat/eval_table_woutPBM.csv'
    else:
        second_best_eval_metrics_fpath = '../out_WithLat/massive_lstm_eval_metrics_2_.npz'
        best_eval_metrics_fpath = '../out_WithLat/avg_lstm_eval_metrics_0_.npz'
        eval_table_fpath = '../out_WithLat/eval_table_withPBM.csv'
    
else:
    if remove_PB:
        second_best_eval_metrics_fpath = '../out/avg_lstm_eval_metrics_4_NoProcessBasedInput_.npz'
        best_eval_metrics_fpath = '../out/massive_lstm_eval_metrics_0_NoProcessBasedInput_.npz'
        eval_table_fpath = '../out/eval_table_woutPBM.csv'
    else:
        second_best_eval_metrics_fpath = '../out/avg_lstm_eval_metrics_3_.npz'
        best_eval_metrics_fpath = '../out/massive_lstm_eval_metrics_1_.npz'
        eval_table_fpath = '../out/eval_table_withPBM.csv'
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
# Objects to store data in
all_train_predictions = np.zeros([n_compare, train_n, seq_len, 1])
all_valid_predictions = np.zeros([n_compare, valid_n, seq_len, 1])
all_valid_loss_values = np.zeros([n_compare, n_epochs])

# Objects to store model info in
model_type_ls = []
model_size_ls = []
model_init_ls = []
model_ran_epochs_ls = [] # for plotting and storing variable len w.r.t. early stopping

count = 0
for model_type in model_names:
    for model_size in pred_fpath_start:
        for i in range(num_init):
            # store relevant info
            model_type_ls.append(model_type)
            model_size_ls.append(model_size)
            model_init_ls.append(i)
            
            # construct the filepaths for both partition's predictions and the associated loss curves
            if remove_PB:
                train_pred_file = extended_dir + train_out_dir + model_size + '_' + model_type + '_train_preds_' + str(i) + '_NoProcessBasedInput_.npy'
                valid_pred_file = extended_dir + train_out_dir + model_size + '_' + model_type + '_valid_preds_' + str(i) + '_NoProcessBasedInput_.npy'
                loss_file = extended_dir + train_out_dir + model_size + '_' + model_type + '_loss_lists_' + str(i) + '_NoProcessBasedInput_.npz'
            else:
                train_pred_file = extended_dir + train_out_dir + model_size + '_' + model_type + '_train_preds_' + str(i) + '_.npy'
                valid_pred_file = extended_dir + train_out_dir + model_size + '_' + model_type + '_valid_preds_' + str(i) + '_.npy'
                loss_file = extended_dir + train_out_dir + model_size + '_' + model_type + '_loss_lists_' + str(i) + '_.npz'      
            
            # load
            train_predictions = np.load(train_pred_file)
            valid_predictions = np.load(valid_pred_file)
            valid_loss_values = np.load(loss_file, allow_pickle=True)['valid_loss']
            
            # update true epochs for this model run w.r.t. early stopping
            ran_epochs = len(valid_loss_values)
            model_ran_epochs_ls.append(ran_epochs)
            
            # store data
            all_train_predictions[count] = train_predictions
            all_valid_predictions[count] = valid_predictions
            all_valid_loss_values[count, :ran_epochs] = valid_loss_values
    
            count += 1
```

```python
# Programmatically identify process-based ice flag data
ice_var_idx = int(np.argwhere(train_variables == 'ice'))
assert valid_variables[ice_var_idx] == 'ice'
```

# View loss curves

```python
fig, ax = plt.subplots(1, 2, figsize = (15, 3))

# Plot all curves
for i in range(n_compare):
    # Label accordingly
    name = model_size_ls[i] + ' ' + model_type_ls[i]
    ax[0].plot(all_valid_loss_values[i][:model_ran_epochs_ls[i]],
               alpha = 0.5, color = color_dict[name])
# Zoom in to the extent that matters
ax[0].set_ylim(0.06, 0.12)
# Further labeling
ax[0].set_title('Relevant Extent of Training Curve - Validation Loss')

# Plot all curves
for i in range(n_compare):
    name = model_size_ls[i] + ' ' + model_type_ls[i]
    ax[1].plot(all_valid_loss_values[i][:model_ran_epochs_ls[i]],
               alpha = 0.5, label = name, color = color_dict[name])
# Handling redundant legend labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
ax[1].legend(by_label.values(), by_label.keys())
# Further labeling
ax[1].set_title('Full Training Curve - Validation Loss');
```

# Start a pandas dataframe to store relevant information in

```python
name_ls = []
min_pred_prob = []
max_pred_prob = []

for i in range(n_compare):
    # Label accordingly
    name = model_size_ls[i] + ' ' + model_type_ls[i]
    cur_valid_predictions = all_valid_predictions[i]
    
    name_ls.append(name)
    min_pred_prob.append(cur_valid_predictions.min())
    max_pred_prob.append(cur_valid_predictions.max())
    
eval_df = pd.DataFrame({'name':name_ls, 'min_pred_prob':min_pred_prob, 'max_pred_prob':max_pred_prob})
eval_df
```

# Convert predicted probabilities into predicted class (ice or not)

```python
all_train_predictions_class = np.round(all_train_predictions)
all_valid_predictions_class = np.round(all_valid_predictions)
```

# Ditch massive transformers

Their poor convergence results in very low predicted probabilities that breaks the rest of the code (i.e., they never predict ice)

```python
n_compare += -5 # get rid of massive transformer results that break code

eval_df = eval_df.iloc[:-5, :].copy()
```

# Overall accuracy

Including quick check on overfitting

```python
def calc_accuracy(pred_probs, obs):
    # Here, the 0-index is just getting rid of the feature
    # dimension of the (originally) pytorch prediction object
    flat_preds = pred_probs[:, :, 0].flatten()
    flat_obs = obs.flatten()
    
    return np.sum(flat_preds == flat_obs) / flat_obs.shape[0]
    
assert train_variables[ice_var_idx] == valid_variables[ice_var_idx] == 'ice'

def calc_accuracy_iceflags(inputs, obs):
    
    flat_ice_flags = inputs[:, :, ice_var_idx].flatten()
    flat_obs = obs.flatten()
    
    return np.sum(flat_ice_flags == flat_obs) / flat_obs.shape[0]
```

```python
# TO DO: add PB ice flag
train_accuracy_ls = []
valid_accuracy_ls = []

# get accuracy for each model
for i in range(n_compare):
    train_accuracy = calc_accuracy(all_train_predictions_class[i], train_y)
    valid_accuracy = calc_accuracy(all_valid_predictions_class[i], valid_y)
    
    train_accuracy_ls.append(train_accuracy)
    valid_accuracy_ls.append(valid_accuracy)

# store accuracy
eval_df['train_accuracy'] = train_accuracy_ls
eval_df['valid_accuracy'] = valid_accuracy_ls
```

# Error of predicted ice on/off dates


### Derive predicted max ice on and max ice off dates

```python
def extract_date_i_and_date(ice_indications, dates):
    
    # ice_on_cutoff is globally defined
    
    ice_on_ids = []
    ice_off_ids = []
    
    ice_on = []
    ice_off = []

    for i in range(ice_indications.shape[0]):
        # Result in +1 when ice emerges and -1 when ice disappears. 0 otherwise.
        diff_i = np.diff(ice_indications[i, :])

        # Indexing by -1 finds the latest/"max" occurrence
        # for ice_on, we need to avoid late season refreeze ice on dates
        ice_on_index = np.argwhere(diff_i == 1)[np.argwhere(diff_i == 1) < ice_on_cutoff][-1].item()
        ice_off_index = np.argwhere(diff_i == -1)[-1].item()

        # Store found indices
        ice_on_ids.append(ice_on_index)
        ice_off_ids.append(ice_off_index)
        
        # Store found dates
        ice_on.append(dates[i, ice_on_index])
        ice_off.append(dates[i, ice_off_index])
        
    return(ice_on_ids, ice_off_ids,
           ice_on, ice_off)
```

```python
# important dates = last ice on and last ice off
# get important dates and date-indices for observations
objects = extract_date_i_and_date(valid_y, valid_dates)
obs_ice_on_ids, obs_ice_off_ids, obs_ice_on, obs_ice_off = objects
```

```python
# get important dates and date-indices for process-based ice flag
objects = extract_date_i_and_date(valid_x[:, :, ice_var_idx], valid_dates)
flag_ice_on_ids, flag_ice_off_ids, flag_ice_on, flag_ice_off = objects
```

```python tags=[]
# get important dates and date-indices for all models
all_pred_ice_on_ids = np.zeros([n_compare, valid_n])
all_pred_ice_off_ids = np.zeros([n_compare, valid_n])
all_pred_ice_on = np.zeros([n_compare, valid_n]).astype(str)
all_pred_ice_off = np.zeros([n_compare, valid_n]).astype(str)

for i in range(n_compare):
    # 11 never predicts about 0.5
    # 13 never predicts melt
    if remove_PB and not use_lat and i in bad_remove_PB_wout_Lat_indices:
        continue
    else:
        objects = extract_date_i_and_date(all_valid_predictions_class[i, :, :, 0], valid_dates)
        pred_ice_on_ids, pred_ice_off_ids, pred_ice_on, pred_ice_off = objects

        all_pred_ice_on_ids[i] = pred_ice_on_ids
        all_pred_ice_off_ids[i] = pred_ice_off_ids
        all_pred_ice_on[i] = pred_ice_on
        all_pred_ice_off[i] = pred_ice_off
```

### Define some functions to deal with those dates

```python
def calc_date_errors(pred_dates, obs_dates):
    # convert obj/str dates into datetimes
    formatted_obs_dates = np.asarray(obs_dates).astype(np.datetime64)
    formatted_pred_dates = np.asarray(pred_dates).astype(np.datetime64)
    
    # calc simple difference
    pred_errors = formatted_obs_dates - formatted_pred_dates
    
    # convert datetime difference to int (days)
    pred_errors = pred_errors.astype(int)
    
    return pred_errors

# simple RMSE calculation
def calc_rmse(errors):
    return np.sqrt(np.sum(errors**2) / len(errors))
```

### Errors for ice on

```python
# calculate all errors: obs - pred
all_pred_ice_on_error = np.zeros([n_compare, valid_n])
all_pred_ice_on_error[:] = np.nan

for i in range(n_compare):
    if remove_PB and not use_lat and i in bad_remove_PB_wout_Lat_indices:
        continue
    else:
        all_pred_ice_on_error[i] = calc_date_errors(all_pred_ice_on[i], obs_ice_on)
    
# PB flag
flag_ice_on_error = calc_date_errors(flag_ice_on, obs_ice_on)
```

```python
# view distribution of errors
plt.hist(flag_ice_on_error, label = 'process-based',
         bins = range(-1*n_days_hist_cutoff, n_days_hist_cutoff), color = 'gray')
plt.legend()
plt.pause(0.001)

for i in range(n_compare):
    name = model_size_ls[i] + ' ' + model_type_ls[i]
    plt.hist(all_pred_ice_on_error[i], 
             bins = range(-1*n_days_hist_cutoff, n_days_hist_cutoff),
             label = name, color = color_dict[name])
# Handling redundant legend labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Error (days)\nobserved - predicted');
```

```python
# calculate average of errors and RMSE
ice_on_avg_error = []
ice_on_rmse = []

for i in range(n_compare):
    
    avg_error = np.mean(all_pred_ice_on_error[i])
    rmse = calc_rmse(all_pred_ice_on_error[i])
    
    ice_on_avg_error.append(avg_error)
    ice_on_rmse.append(rmse)

# store in dataframe
eval_df['ice_on_avg_error'] = ice_on_avg_error
eval_df['ice_on_rmse'] = ice_on_rmse
```

### Errors for ice off

```python
# calculate all errors: obs - pred
all_pred_ice_off_error = np.zeros([n_compare, valid_n])
all_pred_ice_off_error[:] = np.nan

for i in range(n_compare):
    if remove_PB and not use_lat and i in bad_remove_PB_wout_Lat_indices:
        continue
    else:
        all_pred_ice_off_error[i] = calc_date_errors(all_pred_ice_off[i], obs_ice_off)
    
# PB flag
flag_ice_off_error = calc_date_errors(flag_ice_off, obs_ice_off)
```

```python
# view distribution of errors
plt.hist(flag_ice_off_error, label = 'process-based',
         bins = range(-1*n_days_hist_cutoff, n_days_hist_cutoff), color = 'gray')
plt.legend()
plt.pause(0.001)

for i in range(n_compare):
    name = model_size_ls[i] + ' ' + model_type_ls[i]
    plt.hist(all_pred_ice_off_error[i], 
             bins = range(-1*n_days_hist_cutoff, n_days_hist_cutoff),
             label = name, color = color_dict[name])
# Handling redundant legend labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xlabel('Error (days)\nobserved - predicted');
```

```python
# calculate average of errors and RMSE
ice_off_avg_error = []
ice_off_rmse = []

for i in range(n_compare):
    if remove_PB and not use_lat and i in bad_remove_PB_wout_Lat_indices:
        ice_off_avg_error.append(np.nan)
        ice_off_rmse.append(np.nan)
    else:
        avg_error = np.mean(all_pred_ice_off_error[i])
        rmse = calc_rmse(all_pred_ice_off_error[i])

        ice_off_avg_error.append(avg_error)
        ice_off_rmse.append(rmse)
    
# store in dataframe
eval_df['ice_off_avg_error'] = ice_off_avg_error
eval_df['ice_off_rmse'] = ice_off_rmse
```

### Error for predicted ice duration

```python
# Simply take number of days between ice on and ice off for ice duration
def calc_ice_duration(off, on):
    dur = np.asarray(off).astype(np.datetime64) - np.asarray(on).astype(np.datetime64)
    return dur.astype(int)
```

```python
ice_dur_avg_error = []
ice_dur_rmse = []

for i in range(n_compare):
    if remove_PB and not use_lat and i in bad_remove_PB_wout_Lat_indices:
        ice_dur_avg_error.append(np.nan)
        ice_dur_rmse.append(np.nan)
    else:
        obs_dur = calc_ice_duration(obs_ice_off, obs_ice_on)
        pred_dur = calc_ice_duration(all_pred_ice_off[i], all_pred_ice_on[i])

        # error
        pred_error_dur = obs_dur - pred_dur

        # average error and rmse
        ice_dur_avg_error.append(np.mean(pred_error_dur))
        ice_dur_rmse.append(calc_rmse(pred_error_dur))
    
# PB ice flag
flag_dur = calc_ice_duration(flag_ice_off, flag_ice_on)
flag_error_dur = obs_dur - flag_dur
    
# store
eval_df['ice_dur_avg_error'] = ice_dur_avg_error
eval_df['ice_dur_rmse'] = ice_dur_rmse
```

# Add process-based ice flag evaluation to the dataframe

```python
new_row = ['process-based', 0, 1,
 calc_accuracy(train_x[:, :, [ice_var_idx]], train_y),
 calc_accuracy(valid_x[:, :, [ice_var_idx]], valid_y),
 np.mean(flag_ice_on_error),
 calc_rmse(flag_ice_on_error),
 np.mean(flag_ice_off_error),
 calc_rmse(flag_ice_off_error),
 np.mean(flag_error_dur),
 calc_rmse(flag_error_dur)]

eval_df.loc[len(eval_df)] = new_row
```

# Find model that improved upon PB baseline the best (and second best)

As determined by average reduction in ice on, ice off, and ice duration RMSE

```python
# get the percent change in RMSEs relative to PB ice flags
rmse_df = eval_df.loc[:, ['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']]
rmse_pb = rmse_df.iloc[-1]
perc_change_rmse = (rmse_df - rmse_pb)/rmse_pb

# average percent change across the 3 columns
eval_df['avg_rmse_change'] = np.mean(perc_change_rmse.values, axis = 1)
```

```python
# rank all model realizations by their avg rmse improvement
temp = eval_df['avg_rmse_change'].values.argsort()
ranks = np.empty_like(temp)
ranks[temp] = np.arange(len(eval_df['avg_rmse_change'].values))
eval_df['avg_rmse_change_rank'] = ranks
```

```python
best_i = np.argsort(eval_df['avg_rmse_change'])[0]
second_best_i = np.argsort(eval_df['avg_rmse_change'])[1]
```

```python
# display full eval_df
eval_df
```

```python
# models to plot (dropping process-based)
focal_models = ['avg lstm', 'large lstm', 'massive lstm', 'avg transformer', 'large transformer']

# creating a new dataframe for easier boxplot plotting
avg_rmse_change_df = pd.DataFrame({'avg lstm':eval_df[eval_df['name'] == 'avg lstm']['avg_rmse_change'].astype(float).values})
# every model is a column
avg_rmse_change_df['large lstm'] = eval_df[eval_df['name'] == 'large lstm']['avg_rmse_change'].astype(float).values
avg_rmse_change_df['massive lstm'] = eval_df[eval_df['name'] == 'massive lstm']['avg_rmse_change'].astype(float).values
avg_rmse_change_df['avg transformer'] = eval_df[eval_df['name'] == 'avg transformer']['avg_rmse_change'].astype(float).values
avg_rmse_change_df['large transformer'] = eval_df[eval_df['name'] == 'large transformer']['avg_rmse_change'].astype(float).values
avg_rmse_change_df = 100*avg_rmse_change_df

# Start plotting
plt.figure(figsize = (6, 6))

# add 0-line reference representing no improvement over process-based
plt.axhline(0, color = 'black', linestyle = '--')
# add boxplot for conveniance, note: low n (5)
box_plot = plt.boxplot(avg_rmse_change_df[focal_models]);
count = 0
for median in box_plot['medians']:
    median.set_color(list(color_dict.values())[count])
    count += 1

# plot individual performances
count = 1
last_name = eval_df['name'][count]
for i in range(n_compare):
    cur_name = eval_df['name'][i]
    if cur_name != last_name:
        count += 1
        last_name = eval_df['name'][i]
    plt.scatter(count, eval_df['avg_rmse_change'][i]*100, color = color_dict[eval_df['name'][i]])
    
# extra formatting
plt.ylabel('Change in RMSE (%)\nRelative to Process-Based RMSE')
plt.title('Average Change in Date-based Performance:\nIce on, Ice off, Ice Duration')
plt.xticks(range(1,6), focal_models, rotation = 45);
#plt.savefig('../../overall_model_comparison.PNG', dpi = 300, bbox_inches = 'tight')
```

```python
# subsetting keeps NAs
per_date_based_improvement = 100*((rmse_df.iloc[:-1].values - rmse_pb.values) / rmse_pb.values)
titles = ['Ice on', 'Ice off', 'Ice duration']

plt.figure(figsize = (6, 6))

# remove NAs before boxplotting, ASSUMES: NA are consistent across columns. Plot will fail if this assumption fails.
plt_objects = plt.boxplot(per_date_based_improvement[~np.isnan(per_date_based_improvement[:, 0])])
plt.axhline(0, color = 'gray', linestyle = '--')
plt.scatter(range(1, len(titles)+1), per_date_based_improvement[best_i],
            color = color_dict[eval_df['name'][best_i]], marker = 'x',
            s = 100, label = 'best model', linewidth = 3)
plt.legend()
plt.ylabel('Change in RMSE (%)\nRelatiuve to Process-based RMSE')

print("\nMedian improvement\t Best improvement\tChosen model's improvement")
for i in range(len(plt_objects['medians'])):
    print(plt_objects['medians'][i].get_xydata()[0, 1], '\t',
          np.nanmin(per_date_based_improvement[:, i]), '\t',
          per_date_based_improvement[best_i, i])

plt.xticks(range(1, len(titles)+1), titles);
```

~The best `massive lstm` performed best, but `massive lstm`s, in general, were a less reliable training set up. Desirable this unreliability across seeds, the best `massive lstm` was the best in 2/3 date-based performance categories (ice off and ice duration); it was near-best on ice on prediction too.~

~The 2nd best model was an `avg lstm`. This set up is common and was more consistent than the `massive lstm`s.~

~Both will be examined with XAI methods.~

~When using latitude and using PBMs, the massive LSTMs appear pretty consistently reliable, but they are in steep and ultimately losing competition with average-sized LSTMs. The transformers are not as competitative and get worse with increasing size.~

When using latitude and not using PBMs, the massive LSTMs appear pretty consistently reliably, an improvement on large LSTMs, and an improvement over avg LSTMs. The transformers are not as competitative and get worse with increasing size.


# Final, subset view of df

```python
eval_df.iloc[[best_i, second_best_i, -1]]
```

# Save these calculcated errors

```python
# Batch all these calculated errors together
evals1 = {'flag_error_ice_on':flag_ice_on_error,
          'flag_error_ice_off':flag_ice_off_error,
          'flag_error_dur':flag_error_dur,
          'pred_error_ice_on':all_pred_ice_on_error[best_i],
          'pred_error_ice_off':all_pred_ice_off_error[best_i],
          'pred_error_dur':obs_dur - calc_ice_duration(all_pred_ice_off[best_i],
                                                       all_pred_ice_on[best_i]),
          'model_name_ls':eval_df['name'][best_i]}
```

```python
np.savez_compressed(best_eval_metrics_fpath, **evals1)
```

```python
# Batch all these calculated errors together
evals2 = {'flag_error_ice_on':flag_ice_on_error,
          'flag_error_ice_off':flag_ice_off_error,
          'flag_error_dur':flag_error_dur,
          'pred_error_ice_on':all_pred_ice_on_error[second_best_i],
          'pred_error_ice_off':all_pred_ice_off_error[second_best_i],
          'pred_error_dur':obs_dur - calc_ice_duration(all_pred_ice_off[second_best_i],
                                                       all_pred_ice_on[second_best_i]),
          'model_name_ls':eval_df['name'][second_best_i]}
```

```python
np.savez_compressed(second_best_eval_metrics_fpath, **evals2)
```

```python
eval_df.to_csv(eval_table_fpath)
```

<br><br><br><br><br>

# Additional/optional: gather all runs

```python
count = 0 
for out_dir in ['../out/', '../out_WithLat/']:
    if out_dir == '../out/':
        lat_flag = 'False'
    else:
        lat_flag = 'True'
    for PBM_usage in ['woutPBM', 'withPBM']:
        if PBM_usage == 'woutPBM':
            pbm_flag = 'False'
        else:
            pbm_flag = 'True'
            
        cur_fpath = out_dir + 'eval_table_' + PBM_usage + '.csv'
        
        cur_eval_df = pd.read_csv(cur_fpath)
        cur_eval_df['lat_flag'] = lat_flag
        cur_eval_df['pbm_flag'] = pbm_flag
        
        if count == 0:
            whole_eval_df = cur_eval_df.copy()
        else:
            whole_eval_df = pd.concat([whole_eval_df, cur_eval_df])
        count += 1
        
whole_eval_df = whole_eval_df[whole_eval_df['name'] != 'process-based']
whole_eval_df = whole_eval_df.reset_index(drop=True)
```

```python
# get the percent change in RMSEs relative to PB ice flags
rmse_df = whole_eval_df.loc[:, ['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']]
rmse_pb = eval_df.iloc[-1][['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']]
perc_change_rmse = (rmse_df - rmse_pb)/rmse_pb
perc_change_rmse = perc_change_rmse.astype(float) # for some reason (df - df / df) -> object
perc_change_rmse
```

```python
perc_change_rmse['ice_on_rmse'].median(), whole_eval_df['ice_on_rmse'].median()
```

```python
perc_change_rmse['ice_off_rmse'].median(), whole_eval_df['ice_off_rmse'].median()
```

```python
perc_change_rmse['ice_dur_rmse'].median(), whole_eval_df['ice_dur_rmse'].median()
```

```python
print(whole_eval_df['ice_on_avg_error'].mean(), whole_eval_df['ice_on_avg_error'].std(), '\n',
     whole_eval_df['ice_off_avg_error'].mean(), whole_eval_df['ice_off_avg_error'].std(), '\n',
     whole_eval_df['ice_dur_avg_error'].mean(), whole_eval_df['ice_dur_avg_error'].std())
```

```python
print(whole_eval_df['ice_on_rmse'].mean(), whole_eval_df['ice_on_rmse'].std(), '\n',
     whole_eval_df['ice_off_rmse'].mean(), whole_eval_df['ice_off_rmse'].std(), '\n',
     whole_eval_df['ice_dur_rmse'].mean(), whole_eval_df['ice_dur_rmse'].std())
```

```python
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx()

for i in range(whole_eval_df.shape[0]):

    x_adjust = 0.25
    if whole_eval_df['lat_flag'][i] == 'True':
        marker = '^'
    else:
        x_adjust += -.125
        marker = 's'
        
    if whole_eval_df['pbm_flag'][i] == 'True':
        x_adjust += -.25
        markercolor = color_dict[whole_eval_df['name'][i]]
    else:
        markercolor = 'None'
    
    ax.scatter(np.argwhere(np.asarray(list(color_dict.values())) == color_dict[whole_eval_df['name'][i]]).item() + x_adjust,
                whole_eval_df['avg_rmse_change'][i],
                edgecolor = color_dict[whole_eval_df['name'][i]],
                color = markercolor,
                marker = marker, zorder = 1)
    ax2.scatter(np.argwhere(np.asarray(list(color_dict.values())) == color_dict[whole_eval_df['name'][i]]).item() + x_adjust,
                whole_eval_df['avg_rmse_change'][i],
                marker = 'None')

ax2.axhline(0, color = 'black', linestyle = '--')
ax.axhline(0, color = 'black', linestyle = '--')
    
new_yticks = []
for perc_change in ax.get_yticks():
    new_yticks.append(perc_change*eval_df.iloc[-1][['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']].mean() + eval_df.iloc[-1][['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']].mean())
ax.set_yticklabels(np.round(new_yticks, 1))

nice_RMSEs = [7, 8, 9, 10, 11, 12]
new_yticks = []
for nice_RMSE in nice_RMSEs:
    new_yticks.append((nice_RMSE - eval_df.iloc[-1][['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']].mean()) / eval_df.iloc[-1][['ice_on_rmse', 'ice_off_rmse', 'ice_dur_rmse']].mean())
ax.set_yticks(new_yticks)
ax.set_yticklabels(nice_RMSEs)
for val in new_yticks:
    ax.axhline(val, color = 'lightgray', linewidth = 1/2)
for val in ax.get_xticks()[1:-1]:
    ax.axvline(val + 0.0675, color = 'lightgray', linewidth = 1/2)
    

new_yticks2 = []
for perc_change in ax2.get_yticks():
    new_yticks2.append(str(int(np.round(perc_change*100))) + '%')
ax2.set_yticklabels(new_yticks2)

ax.set_xticks(range(5), ['Average\nLSTM', 'Large\nLSTM', 'Massive\nLSTM', 'Average\nTransformer', 'Large\nTransformer'])

plt.title('Average date-based performance of models')
ax2.set_ylabel("Relative to GLM ice flags (%)")
ax.set_ylabel("RMSE (days)")

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [Line2D([0], [0], marker='^', color='None', label='With latitude, without GLM',
                          markerfacecolor='None', markeredgecolor='black'),
                   Line2D([0], [0], marker='s', color='None', label='Without latitude, without GLM',
                          markerfacecolor='None', markeredgecolor='black'),
                   Line2D([0], [0], marker='^', color='None', label='With latitude, with GLM',
                          markerfacecolor='black', markeredgecolor='black'),
                   Line2D([0], [0], marker='s', color='None', label='Without latitude, with GLM',
                          markerfacecolor='black', markeredgecolor='black'),
                   Line2D([0], [0], marker='s', color='black', label='GLM reference', linestyle = '--',
                          markerfacecolor='None', markeredgecolor='None'),
                   Line2D([0], [0], marker='s', color='black', label='Median', linewidth = 3,
                          markerfacecolor='None', markeredgecolor='None')]
ax.legend(handles=legend_elements, bbox_to_anchor = [1.125, 1])

for count, name in enumerate(focal_models):
    x_adjust = 0.25
    for lat_flag in ['True', 'False']:
        for pbm_flag in ['True', 'False']:  
            x_adjust = 0.25
            if lat_flag != 'True':
                x_adjust += -.125
            if pbm_flag == 'True':
                x_adjust += -.25
                
            median_val = whole_eval_df[(whole_eval_df['name'] == name)*(whole_eval_df['lat_flag'] == lat_flag)*(whole_eval_df['pbm_flag'] == pbm_flag)]['avg_rmse_change'].median()
            ax.plot([count+x_adjust-0.0675, count+x_adjust+0.0675], [median_val, median_val], color = 'black', linewidth = 3, zorder = 2)
plt.savefig("../../overall_performance_figure.png", dpi = 300, bbox_inches = 'tight')
```

<br><br><br><br><br>

# Additional recommended plots

I'm doing these in the evaluation metrics notebook instead of the plotting notebook because they are related to higher level prediction targets considered here, rather than evaluating residuals, which is the primary focus of the other notebook.


### Scatter plots of observations versus predictions

```python
fig, ax = plt.subplots(1, 3, figsize = (15, 5))

# Determine a shared lower and upper lim for 1:1 plot
lower_lim = min([min(obs_ice_on_ids),
                 min(flag_ice_on_ids), 
                 min(all_pred_ice_on_ids[best_i]),
                 min(all_pred_ice_on_ids[second_best_i])]) - 5
upper_lim = max([max(obs_ice_on_ids),
                 max(flag_ice_on_ids),
                 max(all_pred_ice_on_ids[best_i]),
                 max(all_pred_ice_on_ids[second_best_i])]) + 5

# Calculate corr
cor_flag = np.round(r2_score(obs_ice_on_ids, flag_ice_on_ids), 3)
cor_pred_sec_best = np.round(r2_score(obs_ice_on_ids, all_pred_ice_on_ids[second_best_i]), 3)
cor_pred_best = np.round(r2_score(obs_ice_on_ids, all_pred_ice_on_ids[best_i]), 3)

# Actual plotting
ax[0].scatter(obs_ice_on_ids, flag_ice_on_ids, alpha = 0.25)
ax[1].scatter(obs_ice_on_ids, all_pred_ice_on_ids[second_best_i], alpha = 0.25)
ax[2].scatter(obs_ice_on_ids, all_pred_ice_on_ids[best_i], alpha = 0.25)

# All the labels
#fig.suptitle('Ice on', fontsize = 14)
ax[0].set_ylabel('Predicted', fontsize = 12)
ax[0].set_title('Process-based Ice Flag\nR-squared = ' + str(cor_flag), fontsize = 12)
ax[1].set_title('Ice on...\nAverage LSTM Prediction\nR-squared = ' + str(cor_pred_sec_best), fontsize = 12)
ax[2].set_title('Massive LSTM Prediction\nR-squared = ' + str(cor_pred_best), fontsize = 12)

# some formating for all plots
for i in range(3):
    ax[i].set_xlabel('Observed\nDay after July 1', fontsize = 12)
    ax[i].plot([lower_lim, upper_lim], [lower_lim, upper_lim], color = 'black')
    ax[i].set_xticks(np.arange(lower_lim, upper_lim, 20))
    ax[i].set_yticks(np.arange(lower_lim, upper_lim, 20));
```

```python
fig, ax = plt.subplots(1, 3, figsize = (15, 5))

# Determine a shared lower and upper lim for 1:1 plot
lower_lim = min([min(obs_ice_off_ids),
                 min(flag_ice_off_ids), 
                 min(all_pred_ice_off_ids[best_i]),
                 min(all_pred_ice_off_ids[second_best_i])]) - 5
upper_lim = max([max(obs_ice_off_ids),
                 max(flag_ice_off_ids),
                 max(all_pred_ice_off_ids[best_i]),
                 max(all_pred_ice_off_ids[second_best_i])]) + 5

# Calculate corr
cor_flag = np.round(r2_score(obs_ice_off_ids, flag_ice_off_ids), 3)
cor_pred_sec_best = np.round(r2_score(obs_ice_off_ids, all_pred_ice_off_ids[second_best_i]), 3)
cor_pred_best = np.round(r2_score(obs_ice_off_ids, all_pred_ice_off_ids[best_i]), 3)

# Actual plotting
ax[0].scatter(obs_ice_off_ids, flag_ice_off_ids, alpha = 0.25)
ax[1].scatter(obs_ice_off_ids, all_pred_ice_off_ids[second_best_i], alpha = 0.25)
ax[2].scatter(obs_ice_off_ids, all_pred_ice_off_ids[best_i], alpha = 0.25)

# All the labels
#fig.suptitle('Ice off', fontsize = 14)
ax[0].set_ylabel('Predicted', fontsize = 12)
ax[0].set_title('Process-based Ice Flag\nR-squared = ' + str(cor_flag), fontsize = 12)
ax[1].set_title('Ice off...\nAverage LSTM Predictioff\nR-squared = ' + str(cor_pred_sec_best), fontsize = 12)
ax[2].set_title('Massive LSTM Predictioff\nR-squared = ' + str(cor_pred_best), fontsize = 12)

# some formating for all plots
for i in range(3):
    ax[i].set_xlabel('Observed\nDay after July 1', fontsize = 12)
    ax[i].plot([lower_lim, upper_lim], [lower_lim, upper_lim], color = 'black')
    ax[i].set_xticks(np.arange(lower_lim, upper_lim, 20))
    ax[i].set_yticks(np.arange(lower_lim, upper_lim, 20));
```

```python
fig, ax = plt.subplots(1, 3, figsize = (15, 5))

second_best_model_ice_dur = calc_ice_duration(all_pred_ice_off[second_best_i],
                                              all_pred_ice_on[second_best_i])
best_model_ice_dur = calc_ice_duration(all_pred_ice_off[best_i],
                                       all_pred_ice_on[best_i])

# Determine a shared lower and upper lim for 1:1 plot
lower_lim = min([min(obs_dur),
                 min(flag_dur), 
                 min(second_best_model_ice_dur),
                 min(best_model_ice_dur)]) - 5
upper_lim = max([max(obs_dur),
                 max(flag_dur),
                 max(second_best_model_ice_dur),
                 max(best_model_ice_dur)]) + 5

# Calculate corr
cor_flag = np.round(r2_score(obs_dur, flag_dur), 3)
cor_pred_sec_best = np.round(r2_score(obs_dur, second_best_model_ice_dur), 3)
cor_pred_best = np.round(r2_score(obs_dur, best_model_ice_dur), 3)

# Actual plotting
ax[0].scatter(obs_dur, flag_dur, alpha = 0.25)
ax[1].scatter(obs_dur, second_best_model_ice_dur, alpha = 0.25)
ax[2].scatter(obs_dur, best_model_ice_dur, alpha = 0.25)

# All the labels
#fig.suptitle('Ice Off', fontsize = 14)
ax[0].set_ylabel('Predicted', fontsize = 12)
ax[0].set_title('Process-based Ice Flag\nR-squared = ' + str(cor_flag), fontsize = 12)
ax[1].set_title('Ice duration...\nAverage LSTM Prediction\nR-squared = ' + str(cor_pred_sec_best), fontsize = 12)
ax[2].set_title('Massive LSTM Prediction\nR-squared = ' + str(cor_pred_best), fontsize = 12)

# some formating for all plots
for i in range(3):
    ax[i].set_xlabel('Observed\nDay after July 1', fontsize = 12)
    ax[i].plot([lower_lim, upper_lim], [lower_lim, upper_lim], color = 'black')
    ax[i].set_xticks(np.arange(lower_lim, upper_lim, 20))
    ax[i].set_yticks(np.arange(lower_lim, upper_lim, 20));
```
