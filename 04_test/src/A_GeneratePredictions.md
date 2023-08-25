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

```python
process_out_dir = '01_process/out/'

soft_test_fpath = process_out_dir + 'soft_test_data.npz'
test_fpath = process_out_dir + 'test_data.npz'


train_out_dir = '02_train/out/'

remove_PB = True

if remove_PB:
    data_scalars_fpath =  train_out_dir + 'massive_lstm_min_max_scalars_0_NoProcessBasedInput_.pt'
    model_weights_fpath = train_out_dir + 'massive_lstm_weights_0_NoProcessBasedInput_.pth'
else:
    data_scalars_fpath =  train_out_dir + 'massive_lstm_min_max_scalars_1_.pt'
    model_weights_fpath = train_out_dir + 'massive_lstm_weights_1_.pth'
```

```python
extended_dir = '/caldera/projects/usgs/water/iidd/datasci/lake-temp/lake_ice_prediction/'

soft_test_fpath = extended_dir + soft_test_fpath
test_fpath = extended_dir + test_fpath

data_scalars_fpath = extended_dir + data_scalars_fpath
model_weights_fpath = extended_dir + model_weights_fpath
```

### Values

```python
# Model hyperparams
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
```

### Outputs

```python
test_out_dir = '04_test/out/'

if remove_PB:
    soft_test_preds_fpath = test_out_dir + 'massive_lstm_soft_test_preds_0_NoProcessBasedInput_.npy'
    test_preds_fpath = test_out_dir + 'massive_lstm_test_preds_0_NoProcessBasedInput_.npy'
else:
    soft_test_preds_fpath = test_out_dir + 'massive_lstm_soft_test_preds_1_.npy'
    test_preds_fpath = test_out_dir + 'massive_lstm_test_preds_1_.npy'
```

```python
soft_test_preds_fpath = extended_dir + soft_test_preds_fpath
test_preds_fpath = extended_dir + test_preds_fpath
```

# Load and prepare data

```python
soft_test = np.load(soft_test_fpath, allow_pickle = True)
test = np.load(test_fpath, allow_pickle = True)
```

```python
soft_test_x =  soft_test['x']
soft_test_variables = soft_test['features']
```

```python
test_x =  test['x']
test_variables = test['features']
```

```python
# Remove the process-based estimate if desired
if remove_PB:
    # remove estimate of ice
    test_ice_loc = np.argwhere(test_variables == 'ice').item()
    soft_test_ice_loc = np.argwhere(soft_test_variables == 'ice').item()
    assert test_ice_loc == soft_test_ice_loc
    test_x = np.delete(test_x, test_ice_loc, -1)
    soft_test_x = np.delete(soft_test_x, test_ice_loc, -1)
    test_variables = np.delete(test_variables, test_ice_loc)
    soft_test_variables = np.delete(soft_test_variables, test_ice_loc)
    
    
    # remove estimate of surface water temp
    test_temp_0_x_loc = np.argwhere(test_variables == 'temp_0_x').item()
    soft_test_temp_0_x_loc = np.argwhere(soft_test_variables == 'temp_0_x').item()
    assert test_temp_0_x_loc == soft_test_temp_0_x_loc
    test_x = np.delete(test_x, test_temp_0_x_loc, -1)
    soft_test_x = np.delete(soft_test_x, test_temp_0_x_loc, -1)
    test_variables = np.delete(test_variables, test_temp_0_x_loc)
    soft_test_variables = np.delete(soft_test_variables, test_temp_0_x_loc)
    
else:
    print('Keeping proces-based estimate')
```

```python
soft_test_x = torch.from_numpy(soft_test_x).float()
test_x = torch.from_numpy(test_x).float()
```

```python
min_max_scalars = torch.load(data_scalars_fpath)

for i in range(soft_test_x.shape[2]):
    # scale soft test set with train min/max
    soft_test_x[:, :, i] = ((soft_test_x[:, :, i] - min_max_scalars[i, 0]) /
                            (min_max_scalars[i, 1] - min_max_scalars[i, 0]))
    
    # scale test set with train min/max
    test_x[:, :, i] = ((test_x[:, :, i] - min_max_scalars[i, 0]) /
                       (min_max_scalars[i, 1] - min_max_scalars[i, 0]))
```

# Define and load model

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
model = BasicLSTM(soft_test_x.shape[2], model_dim, nlayers, dropout_val).cuda()
```

```python
model.load_state_dict(torch.load(model_weights_fpath)) 
```

# Generate predictions

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
soft_test_y_hat = generate_all_preds_via_batch(soft_test_x, bs)
test_y_hat = generate_all_preds_via_batch(test_x, bs)
```

```python
np.save(soft_test_preds_fpath, soft_test_y_hat.numpy())
np.save(test_preds_fpath, test_y_hat.numpy())
```

```python

```
