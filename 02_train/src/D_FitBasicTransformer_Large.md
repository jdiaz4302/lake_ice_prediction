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
from torch import Tensor
import math
import gc
```

# Configuration


### Inputs

```python
process_out_dir = '01_process/out/'

train_data_fpath = process_out_dir + 'train_data.npz'
valid_data_fpath = process_out_dir + 'valid_data.npz'
# not doing any test set stuff until the very, very end
```

```python
extended_dir = '/caldera/projects/usgs/water/iidd/datasci/lake-temp/lake_ice_prediction/'

process_out_dir = extended_dir + process_out_dir

train_data_fpath = extended_dir + train_data_fpath
valid_data_fpath = extended_dir + valid_data_fpath
```

### Values

```python
# 10000 is same epochs as first LSTM
epochs = 10000
# different, coarser printing compared to other models that
# early stop much sooner
coarse_epoch_printing = 50

# model hyperparams
random_seed = 4 # change for different 'random' initializations
model_dim = 512
ff_dim = 2048
nheads = 8

# data loader hyperparams
bs = 375
shuffle = True
pin_memory = True # supposedly faster for CPU->GPU transfers

# training hyperparams
early_stop_patience = 50
```

### Outputs

```python
train_out_dir = '02_train/out/'

# note that file names are adjusted with seed value
data_scalars_fpath =  train_out_dir + 'large_transformer_min_max_scalars_' + str(random_seed) + '_.pt'
model_weights_fpath = train_out_dir + 'large_transformer_weights_' + str(random_seed) + '_.pth'
train_predictions_fpath = train_out_dir + 'large_transformer_train_preds_' + str(random_seed) + '_.npy'
valid_predictions_fpath = train_out_dir + 'large_transformer_valid_preds_' + str(random_seed) + '_.npy'
loss_lists_fpath = train_out_dir + 'large_transformer_loss_lists_' + str(random_seed) + '_.npz'
```

```python
data_scalars_fpath = extended_dir + data_scalars_fpath
model_weights_fpath = extended_dir + model_weights_fpath
train_predictions_fpath = extended_dir + train_predictions_fpath
valid_predictions_fpath = extended_dir + valid_predictions_fpath
loss_lists_fpath = extended_dir + loss_lists_fpath
```

# Import

```python
train_data = np.load(train_data_fpath, allow_pickle = True)
valid_data = np.load(valid_data_fpath, allow_pickle = True)
```

```python
train_x = train_data['x']
train_y = train_data['y']
train_dates = train_data['dates']
train_DOW = train_data['DOW']
train_variables = train_data['features']
```

```python
valid_x = valid_data['x']
valid_y = valid_data['y']
valid_dates = valid_data['dates']
valid_DOW = valid_data['DOW']
valid_variables = valid_data['features']
```

# Prepare data for `torch`

```python
train_y = torch.from_numpy(train_y).float().unsqueeze(2) # adding a feature dimension to Ys
train_x = torch.from_numpy(train_x).float()

valid_y = torch.from_numpy(valid_y).float().unsqueeze(2)
valid_x = torch.from_numpy(valid_x).float()
```

# min-max scale the data

```python
min_max_scalars = torch.zeros(train_x.shape[2], 2)

for i in range(train_x.shape[2]):
    min_max_scalars[i, 0] = train_x[:, :, i].min()
    min_max_scalars[i, 1] = train_x[:, :, i].max()
```

```python
for i in range(train_x.shape[2]):
    # scale train set with train min/max
    train_x[:, :, i] = ((train_x[:, :, i] - min_max_scalars[i, 0]) /
                        (min_max_scalars[i, 1] - min_max_scalars[i, 0]))
    # scale valid set with train min/max
    valid_x[:, :, i] = ((valid_x[:, :, i] - min_max_scalars[i, 0]) /
                        (min_max_scalars[i, 1] - min_max_scalars[i, 0]))
```

# Define a simple transformer (encoder-only)

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 365):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = torch.moveaxis(pe, [0, 1, 2], [1, 0, 2]) # BATCH FIRST
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
```

```python
# recycled model code
class BasicTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, model_dim, nheads, ff_dim):
        super().__init__()
        
        self.embedding = nn.Sequential(
            torch.nn.Linear(input_dim, model_dim),
            torch.nn.ReLU()
        )
        self.pos_encoding = PositionalEncoding(model_dim)
        
        self.mask = (torch.triu(torch.ones((seq_len, seq_len))) == 0)
        self.mask = self.mask.transpose(0, 1)
        self.mask = self.mask.cuda()
        
        self.encoder = nn.TransformerEncoderLayer(d_model = model_dim,
                                                  nhead = nheads,
                                                  batch_first = True,
                                                  dim_feedforward = ff_dim)
        self.dense = nn.Linear(model_dim, 1)
        self.dense_activation = nn.Sigmoid()
        
    def forward(self, x):
        """Assumes x is of shape (batch, sequence, feature)"""
        
        embed = self.embedding(x)
        positioned_embed = self.pos_encoding(embed)
        encoded = self.encoder(positioned_embed, src_mask = self.mask)
        out = self.dense_activation(self.dense(encoded))
        
        return out
```

```python
torch.cuda.is_available(), torch.cuda.device_count()
```

```python
# initialize the model with a seed
torch.manual_seed(random_seed)
# initialize random mini batches with numpy seed
np.random.seed(random_seed)

model = BasicTransformer(train_x.shape[2], train_x.shape[1], model_dim, nheads, ff_dim).cuda()

# print number of model params
sum(param.numel() for param in model.parameters())
```

```python
loss_fn = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters())
loss_ls = []
valid_loss_ls = []
```

```python
train_dataset = []
for i in range(len(train_x)):
    train_dataset.append([train_x[i], train_y[i]])
    
valid_dataset = []
for i in range(len(valid_x)):
    valid_dataset.append([valid_x[i], valid_y[i]])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, pin_memory=pin_memory)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=bs, shuffle=shuffle, pin_memory=pin_memory)
```

```python tags=[]
%%time
worsened_valid_loss_count = 0

for i in range(epochs):
    # for each epoch, perform multiple training updates with the random mini batches of the whole training set
    cur_loss = 0
    for batch_x, batch_y in train_loader:
        # Move data to GPU
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        # Predict and compute loss
        batch_y_hat = model(batch_x)
        batch_loss = loss_fn(batch_y_hat, batch_y)
        # Improve model
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # Track set-wide loss
        cur_loss += batch_loss.item() * batch_x.shape[0]/train_x.shape[0]
        
    # Likewise, generate predictions for the entire validation set
    cur_val_loss = 0
    # No gradient to maximize hardware use (and not needed for validation)
    with torch.no_grad():
        # For all random mini batches in the validation set...
        for batch_val_x, batch_val_y in valid_loader:
            # Again, move to GPU
            batch_val_x, batch_val_y = batch_val_x.cuda(), batch_val_y.cuda()
            # Predict and eval loss
            batch_val_y_hat = model(batch_val_x)
            batch_val_loss = loss_fn(batch_val_y_hat, batch_val_y)
            # Track set-wide loss
            cur_val_loss += batch_val_loss.item() * batch_val_x.shape[0]/valid_x.shape[0]
    
    # Store new set-wide losses
    loss_ls.append(cur_loss)
    valid_loss_ls.append(cur_val_loss)
    
    # Early stopping: determine if validation set performance is degrading
    if cur_val_loss > min(valid_loss_ls):
        worsened_valid_loss_count += 1
        # Break after our patience has been exhausted
        if worsened_valid_loss_count == early_stop_patience:
            break
    # Only save model weights if validation set performance is improving
    else:
        worsened_valid_loss_count = 0
        torch.save(model.state_dict(), model_weights_fpath)
        
    # Occasionally print the current state
    if i % coarse_epoch_printing == 0:
        # epoch, current train loss, current valid loss, best valid loss
        print(i, batch_loss.item(), cur_val_loss, min(valid_loss_ls))
        
# Final print of the current state
print(i, cur_loss, cur_val_loss, min(valid_loss_ls))
```

# Save stuff

```python
batch_loss = None
batch_x = None
batch_y = None
batch_y_hat = None

gc.collect()

torch.cuda.empty_cache()
```

```python
# Reload the best weights and stop performing dropout
model.load_state_dict(torch.load(model_weights_fpath))
model.eval()
```

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
train_y_hat = generate_all_preds_via_batch(train_x, bs)
valid_y_hat = generate_all_preds_via_batch(valid_x, bs)
```

```python
train_y_hat = train_y_hat.numpy()
valid_y_hat = valid_y_hat.numpy()
```

```python
np.save(train_predictions_fpath, train_y_hat)
np.save(valid_predictions_fpath, valid_y_hat)
```

```python
torch.save(min_max_scalars, data_scalars_fpath)
torch.save(model.state_dict(), model_weights_fpath)
```

```python
data = {'train_loss':loss_ls, 'valid_loss':valid_loss_ls}
np.savez_compressed(loss_lists_fpath, **data)
```

```python

```
