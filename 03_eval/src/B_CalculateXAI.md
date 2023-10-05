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
import torch
import torch.nn as nn
```

# Configuration


### Inputs

<!-- #raw -->
# OR THIS
process_out_dir = '01_process/out_WithLat/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_out_dir = '02_train/out_WithLat/'

data_scalars_fpath =  train_out_dir + 'avg_lstm_min_max_scalars_0_.pt'
model_weights_fpath = train_out_dir + 'avg_lstm_weights_0_.pth'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
process_out_dir = '01_process/out_WithLat/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_out_dir = '02_train/out_WithLat/'

data_scalars_fpath =  train_out_dir + 'massive_lstm_min_max_scalars_2_.pt'
model_weights_fpath = train_out_dir + 'massive_lstm_weights_2_.pth'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
process_out_dir = '01_process/out_WithLat/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_out_dir = '02_train/out_WithLat/'

data_scalars_fpath =  train_out_dir + 'massive_lstm_min_max_scalars_3_NoProcessBasedInput_.pt'
model_weights_fpath = train_out_dir + 'massive_lstm_weights_3_NoProcessBasedInput_.pth'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
process_out_dir = '01_process/out_WithLat/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_out_dir = '02_train/out_WithLat/'

data_scalars_fpath =  train_out_dir + 'large_lstm_min_max_scalars_3_NoProcessBasedInput_.pt'
model_weights_fpath = train_out_dir + 'large_lstm_weights_3_NoProcessBasedInput_.pth'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
process_out_dir = '01_process/out_WithLat/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_out_dir = '02_train/out_WithLat/'

data_scalars_fpath =  train_out_dir + 'avg_lstm_min_max_scalars_1_NoProcessBasedInput_.pt'
model_weights_fpath = train_out_dir + 'avg_lstm_weights_1_NoProcessBasedInput_.pth'
<!-- #endraw -->

```python
# OR THIS
process_out_dir = '01_process/out_WithLat/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'

train_out_dir = '02_train/out_WithLat/'

data_scalars_fpath =  train_out_dir + 'large_lstm_min_max_scalars_4_.pt'
model_weights_fpath = train_out_dir + 'large_lstm_weights_4_.pth'
```

```python
extended_dir = '/caldera/projects/usgs/water/iidd/datasci/lake-temp/lake_ice_prediction/'

train_data_fpath = extended_dir + train_data_fpath
valid_data_fpath = extended_dir + valid_data_fpath
data_scalars_fpath = extended_dir + data_scalars_fpath
model_weights_fpath = extended_dir + model_weights_fpath
```

### Values

```python
if 'massive_lstm' in data_scalars_fpath:
    params = 18920961 # matching the full size, encoder only, transformer
    model_dim = int(np.round((1/88)*(np.sqrt(176*params + 4585) - 69))) # assumes 11 variables 
    # ^ solves for y where y = 4x^2 + 49x + 1
    # which was originally y = 11*4*x + 4*x*x + 4*x + 1*x + 1
    dropout_val = 0.1 # matching encoder default value
    nlayers = 6
    bs = 375
elif 'avg_lstm' in data_scalars_fpath:
    model_dim = 16
    dropout_val = 0.1 # matching encoder default value
    nlayers = 1
    bs = 5000
elif 'large_lstm' in data_scalars_fpath:
    params = 3159041 # matching the full size, encoder only, transformer
    model_dim = int(np.round((1/8)*(np.sqrt(16*params + 2385) - 49))) # assumes 11 variables 
    # ^ solves for y where y = 4x^2 + 49x + 1
    # which was originally y = 11*4*x + 4*x*x + 4*x + 1*x + 1
    dropout_val = 0.1 # matching encoder default value
    nlayers = 1
    bs = 1000

    
# when deriving the max ice on date, one detail is that
# we must omit the late ice on dates that occur during a
# (predicted) rethaw. This value is a temporal index
# representing the maximum day after July 1 that can
# be a considered a max ice on date.
# A value of 215 equates to February 1
ice_on_cutoff = 215

eval_seed = 0
### FOR METHOD 1: Expected Gradients ###
# Number of validation samples to calculate EG for
n_eg = 50
# Number of validation samples for finer eval
# (looking at EG with temporal focus on transition dates)
n_eg_fine = 50
# Number of EG algorithm samples (per call to that funct)
eg_samples = 200

### FOR METHOD 2: Permutation-based ###
# Number of times to scramble the data for all validation samples
perm_samples = 200

### FOR METHOD 3: Individual conditional expectation ###
resolution =  25
# impose physical constraints (for out-of-bounds considerations)
# leaving depth and area out because they're log-transformed and seeming non-problematic as-is
vars_to_cap_at_0 = ['ShortWave', 'LongWave', 'RelHum', 'WindSpeed', 'Rain',
                    'Snow', 'ice']
vars_to_cap_at_1 = ['ice']
vars_to_cap_at_100 = ['RelHum']

# remove process-based or not
remove_PB = False
```

### Outputs

<!-- #raw -->
# OR THIS
eval_out_dir = '03_eval/out_WithLat/'

rand_valid_set_EGs_fpath = eval_out_dir + 'avg_lstm_random_valid_eg_coarse_0_.npz'
rand_valid_ice_on_EGs_fpath = eval_out_dir + 'avg_lstm_random_valid_eg_ice_on_0_.npz'
rand_valid_ice_off_EGs_fpath = eval_out_dir + 'avg_lstm_random_valid_eg_ice_off_0_.npz'

valid_set_permutation_fpath = eval_out_dir + 'avg_lstm_permutation_results_0_.npy'

valid_set_ICE_vals_fpath = eval_out_dir + 'avg_lstm_valid_ICE_vals_0_.npy'
valid_set_ICE_preds_fpath = eval_out_dir + 'avg_lstm_valid_ICE_preds_0_.npy'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
eval_out_dir = '03_eval/out_WithLat/'

rand_valid_set_EGs_fpath = eval_out_dir + 'massive_lstm_random_valid_eg_coarse_2_.npz'
rand_valid_ice_on_EGs_fpath = eval_out_dir + 'massive_lstm_random_valid_eg_ice_on_2_.npz'
rand_valid_ice_off_EGs_fpath = eval_out_dir + 'massive_lstm_random_valid_eg_ice_off_2_.npz'

valid_set_permutation_fpath = eval_out_dir + 'massive_lstm_permutation_results_2_.npy'

valid_set_ICE_vals_fpath = eval_out_dir + 'massive_lstm_valid_ICE_vals_2_.npy'
valid_set_ICE_preds_fpath = eval_out_dir + 'massive_lstm_valid_ICE_preds_2_.npy'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
eval_out_dir = '03_eval/out_WithLat/'

rand_valid_set_EGs_fpath = eval_out_dir + 'massive_lstm_random_valid_eg_coarse_3_NoProcessBasedInput_.npz'
rand_valid_ice_on_EGs_fpath = eval_out_dir + 'massive_lstm_random_valid_eg_ice_on_3_NoProcessBasedInput_.npz'
rand_valid_ice_off_EGs_fpath = eval_out_dir + 'massive_lstm_random_valid_eg_ice_off_3_NoProcessBasedInput_.npz'

valid_set_permutation_fpath = eval_out_dir + 'massive_lstm_permutation_results_3_NoProcessBasedInput_.npy'

valid_set_ICE_vals_fpath = eval_out_dir + 'massive_lstm_valid_ICE_vals_3_NoProcessBasedInput_.npy'
valid_set_ICE_preds_fpath = eval_out_dir + 'massive_lstm_valid_ICE_preds_3_NoProcessBasedInput_.npy'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
eval_out_dir = '03_eval/out_WithLat/'

rand_valid_set_EGs_fpath = eval_out_dir + 'large_lstm_random_valid_eg_coarse_3_NoProcessBasedInput_.npz'
rand_valid_ice_on_EGs_fpath = eval_out_dir + 'large_lstm_random_valid_eg_ice_on_3_NoProcessBasedInput_.npz'
rand_valid_ice_off_EGs_fpath = eval_out_dir + 'large_lstm_random_valid_eg_ice_off_3_NoProcessBasedInput_.npz'

valid_set_permutation_fpath = eval_out_dir + 'large_lstm_permutation_results_3_NoProcessBasedInput_.npy'

valid_set_ICE_vals_fpath = eval_out_dir + 'large_lstm_valid_ICE_vals_3_NoProcessBasedInput_.npy'
valid_set_ICE_preds_fpath = eval_out_dir + 'large_lstm_valid_ICE_preds_3_NoProcessBasedInput_.npy'
<!-- #endraw -->

<!-- #raw -->
# OR THIS
eval_out_dir = '03_eval/out_WithLat/'

rand_valid_set_EGs_fpath = eval_out_dir + 'avg_lstm_random_valid_eg_coarse_1_NoProcessBasedInput_.npz'
rand_valid_ice_on_EGs_fpath = eval_out_dir + 'avg_lstm_random_valid_eg_ice_on_1_NoProcessBasedInput_.npz'
rand_valid_ice_off_EGs_fpath = eval_out_dir + 'avg_lstm_random_valid_eg_ice_off_1_NoProcessBasedInput_.npz'

valid_set_permutation_fpath = eval_out_dir + 'avg_lstm_permutation_results_1_NoProcessBasedInput_.npy'

valid_set_ICE_vals_fpath = eval_out_dir + 'avg_lstm_valid_ICE_vals_1_NoProcessBasedInput_.npy'
valid_set_ICE_preds_fpath = eval_out_dir + 'avg_lstm_valid_ICE_preds_1_NoProcessBasedInput_.npy'
<!-- #endraw -->

```python
# OR THIS
eval_out_dir = '03_eval/out_WithLat/'

rand_valid_set_EGs_fpath = eval_out_dir + 'large_lstm_random_valid_eg_coarse_4_.npz'
rand_valid_ice_on_EGs_fpath = eval_out_dir + 'large_lstm_random_valid_eg_ice_on_4_.npz'
rand_valid_ice_off_EGs_fpath = eval_out_dir + 'large_lstm_random_valid_eg_ice_off_4_.npz'

valid_set_permutation_fpath = eval_out_dir + 'large_lstm_permutation_results_4_.npy'

valid_set_ICE_vals_fpath = eval_out_dir + 'large_lstm_valid_ICE_vals_4_.npy'
valid_set_ICE_preds_fpath = eval_out_dir + 'large_lstm_valid_ICE_preds_4_.npy'
```

```python
rand_valid_set_EGs_fpath = extended_dir + rand_valid_set_EGs_fpath
rand_valid_ice_on_EGs_fpath = extended_dir + rand_valid_ice_on_EGs_fpath
rand_valid_ice_off_EGs_fpath = extended_dir + rand_valid_ice_off_EGs_fpath

valid_set_permutation_fpath = extended_dir + valid_set_permutation_fpath

valid_set_ICE_vals_fpath = extended_dir + valid_set_ICE_vals_fpath
valid_set_ICE_preds_fpath = extended_dir + valid_set_ICE_preds_fpath
```

# Quick check that files match up

```python
# lump all files together
files = [data_scalars_fpath, model_weights_fpath,
         rand_valid_set_EGs_fpath, rand_valid_ice_on_EGs_fpath,
         rand_valid_ice_off_EGs_fpath, valid_set_permutation_fpath,
         valid_set_ICE_vals_fpath, valid_set_ICE_preds_fpath]

# extract their specified size and seed value
file_model_sizes = []
file_model_seeds = []
if remove_PB:
    for file in files:
        file_model_sizes.append(files[0].split("/")[-1].split("_")[0])
        file_model_seeds.append(file.split('_')[-3])
        
else:
    for file in files:
        file_model_sizes.append(file.split('_')[3].split('/')[-1])
        file_model_seeds.append(file.split('_')[-2])
    
# make sure only 1 unique size and seed exists among files
assert len(np.unique(np.asarray(file_model_sizes))) == 1
assert len(np.unique(np.asarray(file_model_seeds))) == 1
```

# Load data

```python
train_data = np.load(train_data_fpath, allow_pickle = True)
valid_data = np.load(valid_data_fpath, allow_pickle = True)
```

```python
train_x = train_data['x']
train_variables = train_data['features']
```

```python
valid_x = valid_data['x']
valid_y = valid_data['y']
valid_variables = valid_data['features']
```

```python
# Remove the process-based estimate if desired
if remove_PB:
    # remove estimate of ice
    train_ice_loc = np.argwhere(train_variables == 'ice').item()
    valid_ice_loc = np.argwhere(valid_variables == 'ice').item()
    assert train_ice_loc == valid_ice_loc
    train_x = np.delete(train_x, train_ice_loc, -1)
    valid_x = np.delete(valid_x, train_ice_loc, -1)
    train_variables = np.delete(train_variables, train_ice_loc)
    valid_variables = np.delete(valid_variables, train_ice_loc)
    
    
    # remove estimate of surface water temp
    train_temp_0_x_loc = np.argwhere(train_variables == 'temp_0_x').item()
    valid_temp_0_x_loc = np.argwhere(valid_variables == 'temp_0_x').item()
    assert train_temp_0_x_loc == valid_temp_0_x_loc
    train_x = np.delete(train_x, train_temp_0_x_loc, -1)
    valid_x = np.delete(valid_x, train_temp_0_x_loc, -1)
    train_variables = np.delete(train_variables, train_temp_0_x_loc)
    valid_variables = np.delete(valid_variables, train_temp_0_x_loc)
    
else:
    print('Keeping proces-based estimate')
```

# Scale and format data

```python
train_x = torch.from_numpy(train_x).float()

valid_y = torch.from_numpy(valid_y).float().unsqueeze(2)
valid_x = torch.from_numpy(valid_x).float()
```

```python
min_max_scalars = torch.load(data_scalars_fpath)

for i in range(train_x.shape[2]):
    # scale train set with train min/max
    train_x[:, :, i] = ((train_x[:, :, i] - min_max_scalars[i, 0]) /
                        (min_max_scalars[i, 1] - min_max_scalars[i, 0]))
    # scale valid set with train min/max
    valid_x[:, :, i] = ((valid_x[:, :, i] - min_max_scalars[i, 0]) /
                        (min_max_scalars[i, 1] - min_max_scalars[i, 0]))
```

# Load trained model (with all vars)

```python
class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, nlayers, dropout):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            num_layers = nlayers,
                            batch_first = True)
        self.dropout = nn.Dropout(p = dropout)
        
        self.dense = nn.Linear(hidden_dim, 1)
        self.activation = nn.Sigmoid()
        
    def forward(self, x):
        
        lstm_out, (h, c) = self.lstm(x)
        
        drop_out = self.dropout(lstm_out)
        
        out = self.activation(self.dense(drop_out))
        
        return out
```

```python
model = BasicLSTM(train_x.shape[2], model_dim, nlayers, dropout_val).cuda()
```

```python
model.load_state_dict(torch.load(model_weights_fpath))
```

# Set up a seeded random generator

```python
RNG = np.random.RandomState(eval_seed)
```

# Set up expected gradients (EG)

```python
def expected_gradients(x, x_set, model, n_samples, rng, dim_0_focus=None, dim_1_focus=None):
    
    # dim_0 corresponds to lakes
    # dim_1 corresponds to time steps

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    n_seq = x_set.shape[0]
    seq_len = x_set.shape[1]
    num_vars = x_set.shape[2]

    for k in range(n_samples):
        # SAMPLE A RANDOM BASELINE INPUT
        rand_seq = rng.choice(n_seq) # rand_time may be more accurate
        baseline_x = x_set[[rand_seq]].to(device)

        # SAMPLE A RANDOM SCALE ALONG THE DIFFERENCE
        scale = rng.uniform()

        # SAME IG CALCULATION
        x_diff = x - baseline_x
        curr_x = baseline_x + scale*x_diff
        if curr_x.requires_grad == False:
            curr_x.requires_grad = True
        model.zero_grad()
        y = model(curr_x)

        # GET GRADIENT
        if dim_0_focus == None and dim_1_focus == None:
            gradients = torch.autograd.grad(y[:, :, :], curr_x, torch.ones_like(y[:, :, :]))
        elif dim_1_focus == None and dim_0_focus != None:
            gradients = torch.autograd.grad(y[dim_0_focus, :, :], curr_x, torch.ones_like(y[dim_0_focus, :, :]))
        elif dim_1_focus != None and dim_0_focus == None:
            gradients = torch.autograd.grad(y[:, dim_1_focus, :], curr_x, torch.ones_like(y[:, dim_1_focus, :]))
        else:
            gradients = torch.autograd.grad(y[dim_0_focus, dim_1_focus, :], curr_x, torch.ones_like(y[dim_0_focus, dim_1_focus, :]))

        if k == 0:
            expected_gradients = x_diff*gradients[0] * 1/n_samples
        else:
            expected_gradients = expected_gradients + ((x_diff*gradients[0]) * 1/n_samples)

    return(expected_gradients.detach().cpu().numpy())
```

# Perform EG

```python
# Array to store expected gradient results
valid_eg_results = np.zeros([n_eg, valid_x.shape[1], valid_x.shape[2]])
# List to store the sampled validation indices
sampled_valid_ids = []
```

```python
%%time
for i in range(n_eg):
    # Pick a random validation sample and record it
    rand_valid_i = RNG.choice(valid_x.shape[0])
    sampled_valid_ids.append(rand_valid_i)
    
    # Calc expected gradients and store them
    eg_vals = expected_gradients(valid_x[[rand_valid_i]].cuda(), train_x, model, eg_samples, RNG)
    valid_eg_results[[i]] = eg_vals
```

# Evaluate EGs w.r.t. ice transitions

```python
# Not using the data loader is simpler for variable Shuffle=True/False
# (and I implemented this prior to using formal data loaders)
def generate_all_preds_via_batch(x_tensor, batch_size):
    # make empty array for predictions
    y_hat_tensor = torch.zeros([x_tensor.shape[0], x_tensor.shape[1], 1])
    
    # until we use all the possible sequential batches...
    count = 1
    loop_max = int(np.ceil(x_tensor.shape[0] / batch_size))
    for i in range(loop_max):
        min_i = (count-1)*bs
        max_i = count*bs
        # generate batch-sized predictions
        if i != (loop_max - 1):
            with torch.no_grad():
                y_hat_tensor[min_i:max_i] = model(x_tensor[min_i:max_i].cuda()).cpu()
        # or remaining-sized predictions
        else:
            with torch.no_grad():
                y_hat_tensor[min_i:] = model(x_tensor[min_i:].cuda()).cpu()
        # update batch count
        count += 1
        
    return y_hat_tensor
```

```python
valid_y_hat = generate_all_preds_via_batch(valid_x, bs)
```

```python
# determine when we predict an ice-free to ice-on transition
diff_valid_y_hat = np.diff(np.round(valid_y_hat), axis = 1)

# objects to store in
transition_ids_ice_on = np.zeros([valid_x.shape[0], 1])
transition_ids_ice_off = np.zeros([valid_x.shape[0], 1])

# loop through all sequences
for i in range(valid_x.shape[0]):
    
    seq_of_interest = diff_valid_y_hat[i].flatten()
    
    # identify last freeze before cut off where spring thawing starts
    ice_on_id =  np.argwhere(seq_of_interest == 1)[np.argwhere(seq_of_interest == 1) < ice_on_cutoff][-1].item()
    # identify last thaw
    ice_off_id =  np.argwhere(seq_of_interest == -1)[-1].item()
    
    transition_ids_ice_on[i] = ice_on_id
    transition_ids_ice_off[i] = ice_off_id
```

### Ice on

```python
# reinitiate random generator for same sequences as above
RNG = np.random.RandomState(eval_seed)
```

```python
# Storage objects
valid_eg_results_ice_on = np.zeros([n_eg_fine, valid_x.shape[1], valid_x.shape[2]])
sampled_valid_ids_ice_on = []
```

```python
%%time
# for a few predictions, look at EG values focused on that predicted transition
for i in range(n_eg_fine):
    # Pick a random validation sample and record it
    rand_valid_i = RNG.choice(valid_x.shape[0])
    sampled_valid_ids_ice_on.append(rand_valid_i)
    
    # Calc expected gradients and store them
    eg_vals = expected_gradients(valid_x[[rand_valid_i]].cuda(), train_x, model, eg_samples, RNG,
                                 dim_1_focus = transition_ids_ice_on[rand_valid_i])
    valid_eg_results_ice_on[[i]] = eg_vals
```

### Ice off

```python
# reinitiate random generator for same sequences as above
RNG = np.random.RandomState(eval_seed)
```

```python
# Storage objects
valid_eg_results_ice_off = np.zeros([n_eg_fine, valid_x.shape[1], valid_x.shape[2]])
sampled_valid_ids_ice_off = []
```

```python
%%time
# for a few predictions, look at EG values focused on that predicted transition
for i in range(n_eg_fine):
    # Pick a random validation sample and record it
    rand_valid_i = RNG.choice(valid_x.shape[0])
    sampled_valid_ids_ice_off.append(rand_valid_i)
    
    # Calc expected gradients and store them
    eg_vals = expected_gradients(valid_x[[rand_valid_i]].cuda(), train_x, model, eg_samples, RNG,
                                 dim_1_focus = transition_ids_ice_off[rand_valid_i])
    valid_eg_results_ice_off[[i]] = eg_vals
```

<!-- #region tags=[] -->
<br><br><br><br><br>

# Permutation
<!-- #endregion -->

```python
loss_fn = torch.nn.BCELoss()
```

```python
valid_y_hat = generate_all_preds_via_batch(valid_x, bs)
base_loss = loss_fn(valid_y_hat, valid_y)
base_loss.item()
```

```python tags=[]
%%time
permutation_results =  np.zeros([perm_samples, len(valid_variables)])

for n_i in range(perm_samples):
    ids = np.arange(valid_x.shape[0])
    # an in-place operation
    RNG.shuffle(ids)

    loss_ls = []
    for var_i in range(len(valid_variables)):
        perm_valid_x = valid_x.clone()
        perm_valid_x[:, :, var_i] = valid_x[ids, :, var_i]
        with torch.no_grad():
            perm_valid_y_hat = generate_all_preds_via_batch(perm_valid_x, bs)
            loss = loss_fn(perm_valid_y_hat, valid_y)
        loss_ls.append(loss.item())
        
    permutation_results[n_i] = loss_ls
```

<br><br><br><br><br>

# ICE

```python
# 'resolution + 3' adds room in the grid of values for
#    - a new extreme min
#    - the existing max (python has exclusive max `range`)
#    - a new extreme max
ICE_x_array = np.zeros([len(valid_variables), resolution + 3])
ICE_pred_array = np.zeros([len(valid_variables), resolution + 3, valid_x.shape[0], valid_x.shape[1], 1])
```

```python
for var_index in range(len(valid_variables)):
    
    # Generate a grid of values to make predictions over for each variable
    # get values to change inputs to
    grid_quantiles = np.arange(0, 1 + 1/resolution, 1/resolution)
    grid_values = np.quantile(train_x[:, :, var_index].numpy().flatten(), grid_quantiles)
    
    # add some out-of-bound, extreme values
    lower_extreme_tail_val = grid_values[0] - (grid_values[1] - grid_values[0])
    upper_extreme_tail_val = grid_values[-1] + (grid_values[-1] - grid_values[-2])
    
    # put everything together
    grid_values = np.insert(grid_values, 0, lower_extreme_tail_val)
    grid_values = np.append(grid_values, upper_extreme_tail_val)
    
    # identify the current variable and possibly impose some physical limits
    cur_var = valid_variables[var_index]
    if cur_var in vars_to_cap_at_0:
        imposed_min = ((0 - min_max_scalars[var_index, 0]) / (min_max_scalars[var_index, 1] - min_max_scalars[var_index, 0]))
        grid_values = np.clip(grid_values, a_min = imposed_min, a_max = None)
    if cur_var in vars_to_cap_at_1:
        imposed_max = ((1 - min_max_scalars[var_index, 0]) / (min_max_scalars[var_index, 1] - min_max_scalars[var_index, 0]))
        grid_values = np.clip(grid_values, a_min = None, a_max = imposed_max)
    if cur_var in vars_to_cap_at_100:
        imposed_max = ((100 - min_max_scalars[var_index, 0]) / (min_max_scalars[var_index, 1] - min_max_scalars[var_index, 0]))
        grid_values = np.clip(grid_values, a_min = None, a_max = imposed_max)
    
    # Generate predictions
    val_count = 0
    for val in grid_values:
        cur_x = valid_x.clone()
        cur_x[:, :, var_index] = torch.as_tensor(val)
        cur_y_hat = generate_all_preds_via_batch(cur_x, bs)
        
        # Store val and pred
        ICE_x_array[var_index, val_count] = val
        ICE_pred_array[var_index, val_count] = cur_y_hat
        val_count += 1
```

<!-- #region tags=[] -->
<br><br><br><br><br>

# Save
<!-- #endregion -->

```python
valid_eg_results_bundled = {'results':valid_eg_results,
                            'ids':sampled_valid_ids}

valid_eg_results_ice_on_bundled = {'results':valid_eg_results_ice_on,
                                   'ids':sampled_valid_ids_ice_on}

valid_eg_results_ice_off_bundled = {'results':valid_eg_results_ice_off,
                                    'ids':sampled_valid_ids_ice_off}
```

```python
np.savez_compressed(rand_valid_set_EGs_fpath, **valid_eg_results_bundled)
np.savez_compressed(rand_valid_ice_on_EGs_fpath, **valid_eg_results_ice_on_bundled)
np.savez_compressed(rand_valid_ice_off_EGs_fpath, **valid_eg_results_ice_off_bundled)

np.save(valid_set_permutation_fpath, permutation_results)

np.save(valid_set_ICE_vals_fpath, ICE_x_array)
np.save(valid_set_ICE_preds_fpath, ICE_pred_array)
```

```python

```
