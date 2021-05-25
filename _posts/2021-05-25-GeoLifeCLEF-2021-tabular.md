# Observation data loading



```
import os
from pathlib import Path
import matplotlib.pyplot as plt

import pandas as pd

from fastai.tabular.all import *

# Change this path to adapt to where you downloaded the data
BASE_PATH = Path("/storage/geolifeclef-2021/")
DATA_PATH = BASE_PATH / "data"

# Create the path to save submission files
SUBMISSION_PATH = Path("submissions")
os.makedirs(SUBMISSION_PATH, exist_ok=True)
```

```
from GLC.metrics import top_30_error_rate
from GLC.metrics import top_k_error_rate_from_sets
from GLC.metrics import predict_top_30_set, predict_top_k_set

from sklearn.ensemble import RandomForestClassifier
```

```
def get_observations(data_path):
    df_fr = pd.read_csv(data_path / "observations" / "observations_fr_train.csv",
                        sep=";", index_col="observation_id")
    df_us = pd.read_csv(data_path / "observations" / "observations_us_train.csv",
                        sep=";", index_col="observation_id")
    
    df = pd.concat((df_fr, df_us))
    
    return df

def get_test_observations(data_path):
    df_fr_test = pd.read_csv(DATA_PATH / "observations" / "observations_fr_test.csv", sep=";",
                             index_col="observation_id")
    df_us_test = pd.read_csv(DATA_PATH / "observations" / "observations_us_test.csv", sep=";",
                             index_col="observation_id")
    
    df_test = pd.concat((df_fr_test, df_us_test))
    
    return df_test
```

Then, we retrieve the train/val split provided:

```
df = get_observations(DATA_PATH)

obs_id_train = df.index[df["subset"] == "train"].values
obs_id_val = df.index[df["subset"] == "val"].values

y_train = df.loc[obs_id_train]["species_id"].values
y_val = df.loc[obs_id_val]["species_id"].values

n_val = len(obs_id_val)
print("Validation set size: {} ({:.1%} of train observations)".format(n_val, n_val / len(df)))
```

    /opt/conda/envs/fastai/lib/python3.8/site-packages/numpy/lib/arraysetops.py:583: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)


    Validation set size: 45446 (2.4% of train observations)




We also load the observation data for the test set:


```
df_test = get_test_observations(DATA_PATH)
obs_id_test = df_test.index

print("Number of observations for testing: {}".format(len(df_test)))

df_test.head()
```

    Number of observations for testing: 42405





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
    <tr>
      <th>observation_id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10782781</th>
      <td>43.601788</td>
      <td>6.940195</td>
    </tr>
    <tr>
      <th>10364138</th>
      <td>46.241711</td>
      <td>0.683586</td>
    </tr>
    <tr>
      <th>10692017</th>
      <td>45.181095</td>
      <td>1.533459</td>
    </tr>
    <tr>
      <th>10222322</th>
      <td>46.938450</td>
      <td>5.298678</td>
    </tr>
    <tr>
      <th>10241950</th>
      <td>45.017433</td>
      <td>0.960736</td>
    </tr>
  </tbody>
</table>
</div>



For submissions, we also need the following mapping to correct a slight misalignment in the test observation ids:

```
df_test_obs_id_mapping = pd.read_csv(BASE_PATH / "test_observation_ids_mapping.csv", sep=";")
df_test_obs_id_mapping.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>observation_id</th>
      <th>Id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10782781</td>
      <td>10782781</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10364138</td>
      <td>10364138</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10692017</td>
      <td>10692017</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10222322</td>
      <td>10222322</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10241950</td>
      <td>10241950</td>
    </tr>
  </tbody>
</table>
</div>




# Sample submission file

In this section, we will demonstrate how to generate the sample submission file provided.

To do so, we will use this function:


```
def generate_submission_file(filename, corrected_observation_ids, s_pred):
    s_pred = [
        " ".join(map(str, pred_set))
        for pred_set in s_pred
    ]
    
    df = pd.DataFrame({
        "Id": corrected_observation_ids,
        "Predicted": s_pred
    })
    df.to_csv(filename, index=False)
```

# Random forest on environmental vectors

A classical approach in ecology is to train Random Forests on environmental vectors.

We show here how to do so using scikit-learn.

We start by loading the environmental vectors:

```
df_env = pd.read_csv(DATA_PATH / "pre-extracted" / "environmental_vectors.csv", sep=";", index_col="observation_id")
```

    /opt/conda/envs/fastai/lib/python3.8/site-packages/numpy/lib/arraysetops.py:583: FutureWarning: elementwise comparison failed; returning scalar instead, but in the future will perform elementwise comparison
      mask |= (ar1 == a)


```
df_env.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bio_1</th>
      <th>bio_2</th>
      <th>bio_3</th>
      <th>bio_4</th>
      <th>bio_5</th>
      <th>bio_6</th>
      <th>bio_7</th>
      <th>bio_8</th>
      <th>bio_9</th>
      <th>bio_10</th>
      <th>...</th>
      <th>bio_18</th>
      <th>bio_19</th>
      <th>bdticm</th>
      <th>bldfie</th>
      <th>cecsol</th>
      <th>clyppt</th>
      <th>orcdrc</th>
      <th>phihox</th>
      <th>sltppt</th>
      <th>sndppt</th>
    </tr>
    <tr>
      <th>observation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10000000</th>
      <td>1.420833</td>
      <td>6.908333</td>
      <td>29.272598</td>
      <td>614.1493</td>
      <td>15.1</td>
      <td>-8.5</td>
      <td>23.600000</td>
      <td>-1.000000</td>
      <td>9.183333</td>
      <td>9.466667</td>
      <td>...</td>
      <td>248.0</td>
      <td>358.0</td>
      <td>2082.0</td>
      <td>988.0</td>
      <td>29.0</td>
      <td>13.0</td>
      <td>63.0</td>
      <td>62.0</td>
      <td>34.0</td>
      <td>53.0</td>
    </tr>
    <tr>
      <th>10000001</th>
      <td>8.837500</td>
      <td>9.858334</td>
      <td>37.771393</td>
      <td>586.8139</td>
      <td>23.8</td>
      <td>-2.3</td>
      <td>26.099998</td>
      <td>6.016667</td>
      <td>16.383333</td>
      <td>16.383333</td>
      <td>...</td>
      <td>226.0</td>
      <td>288.0</td>
      <td>1816.0</td>
      <td>1142.0</td>
      <td>20.0</td>
      <td>22.0</td>
      <td>39.0</td>
      <td>58.0</td>
      <td>41.0</td>
      <td>36.0</td>
    </tr>
    <tr>
      <th>10000002</th>
      <td>6.241667</td>
      <td>8.350000</td>
      <td>32.239384</td>
      <td>632.8609</td>
      <td>21.0</td>
      <td>-4.9</td>
      <td>25.900000</td>
      <td>3.033333</td>
      <td>14.200000</td>
      <td>14.200000</td>
      <td>...</td>
      <td>268.0</td>
      <td>317.0</td>
      <td>1346.0</td>
      <td>1075.0</td>
      <td>29.0</td>
      <td>22.0</td>
      <td>54.0</td>
      <td>59.0</td>
      <td>40.0</td>
      <td>38.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 27 columns</p>
</div>



# fast.ai Tabular

```
df['species_id'] = df['species_id'].astype('str')
```

```
data_set = df.merge(df_env, on='observation_id')
```

```
data_set.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>latitude</th>
      <th>longitude</th>
      <th>species_id</th>
      <th>subset</th>
      <th>bio_1</th>
      <th>bio_2</th>
      <th>bio_3</th>
      <th>bio_4</th>
      <th>bio_5</th>
      <th>bio_6</th>
      <th>...</th>
      <th>bio_18</th>
      <th>bio_19</th>
      <th>bdticm</th>
      <th>bldfie</th>
      <th>cecsol</th>
      <th>clyppt</th>
      <th>orcdrc</th>
      <th>phihox</th>
      <th>sltppt</th>
      <th>sndppt</th>
    </tr>
    <tr>
      <th>observation_id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10561949</th>
      <td>45.705116</td>
      <td>1.424622</td>
      <td>241</td>
      <td>train</td>
      <td>11.229167</td>
      <td>8.724999</td>
      <td>37.286324</td>
      <td>556.81506</td>
      <td>24.5</td>
      <td>1.1</td>
      <td>...</td>
      <td>211.0</td>
      <td>287.0</td>
      <td>1678.0</td>
      <td>1381.0</td>
      <td>13.0</td>
      <td>20.0</td>
      <td>26.0</td>
      <td>58.0</td>
      <td>36.0</td>
      <td>44.0</td>
    </tr>
    <tr>
      <th>10131188</th>
      <td>45.146973</td>
      <td>6.416794</td>
      <td>101</td>
      <td>train</td>
      <td>4.587500</td>
      <td>9.058333</td>
      <td>33.302696</td>
      <td>664.60220</td>
      <td>19.9</td>
      <td>-7.3</td>
      <td>...</td>
      <td>265.0</td>
      <td>362.0</td>
      <td>1771.0</td>
      <td>1219.0</td>
      <td>28.0</td>
      <td>18.0</td>
      <td>49.0</td>
      <td>61.0</td>
      <td>38.0</td>
      <td>45.0</td>
    </tr>
    <tr>
      <th>10076047</th>
      <td>49.746944</td>
      <td>4.686389</td>
      <td>38</td>
      <td>train</td>
      <td>9.670834</td>
      <td>8.608334</td>
      <td>36.169468</td>
      <td>571.84100</td>
      <td>23.2</td>
      <td>-0.6</td>
      <td>...</td>
      <td>227.0</td>
      <td>244.0</td>
      <td>1980.0</td>
      <td>1377.0</td>
      <td>19.0</td>
      <td>29.0</td>
      <td>29.0</td>
      <td>65.0</td>
      <td>46.0</td>
      <td>24.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



```
data_set.drop_duplicates(inplace=True)
```

```
value_counts = data_set['species_id'].value_counts()
```

```
data_set = data_set[data_set.species_id.isin(value_counts[value_counts > 90].index)]
```

```
vocab = list(data_set.species_id.unique())
```

```
len(vocab)
```




    3313



```
y = pd.get_dummies(data_set.species_id)
```

```
data_set = data_set.merge(y, on='observation_id')
```

```
data_set = data_set.drop(columns='species_id')
```

```
del y
```

```
len(data_set)
```




    1453487



```
batch_size = 100_000

def group_labels(df, group_cols, batch_size):
    data_set_batches = []

    for b in range(0, len(df), batch_size):
        batch = df[b:(b + batch_size)]
        batch = batch.groupby(group_cols, dropna=False)[vocab].max().reset_index()
        data_set_batches.append(batch)
        print(f'Batch: {b}, size {len(batch)}')
    
    return pd.concat(data_set_batches)
```

```
data_set = data_set.sample(frac = 1)
```

```
data_set = group_labels(data_set,
                        list(df_env.columns) + ['latitude', 'longitude', 'subset'],
                        batch_size)
```

    Batch: 0, size 99519
    Batch: 100000, size 99523
    Batch: 200000, size 99523
    Batch: 300000, size 99538
    Batch: 400000, size 99571
    Batch: 500000, size 99509
    Batch: 600000, size 99510
    Batch: 700000, size 99567
    Batch: 800000, size 99541
    Batch: 900000, size 99503
    Batch: 1000000, size 99544
    Batch: 1100000, size 99500
    Batch: 1200000, size 99539
    Batch: 1300000, size 99563
    Batch: 1400000, size 53355


```
data_set = data_set.sample(frac = 1)
```

```
data_set = group_labels(data_set,
                        list(df_env.columns) + ['latitude', 'longitude', 'subset'],
                        batch_size)
```

    Batch: 0, size 99745
    Batch: 100000, size 99708
    Batch: 200000, size 99782
    Batch: 300000, size 99744
    Batch: 400000, size 99733
    Batch: 500000, size 99723
    Batch: 600000, size 99720
    Batch: 700000, size 99719
    Batch: 800000, size 99715
    Batch: 900000, size 99742
    Batch: 1000000, size 99737
    Batch: 1100000, size 99735
    Batch: 1200000, size 99745
    Batch: 1300000, size 99740
    Batch: 1400000, size 46738


```
data_set = data_set.sample(frac = 1)
```

```
batch_size = 100_000
data_set = group_labels(data_set,
                        list(df_env.columns) + ['latitude', 'longitude', 'subset'],
                        batch_size)
```

    Batch: 0, size 99775
    Batch: 100000, size 99765
    Batch: 200000, size 99785
    Batch: 300000, size 99763
    Batch: 400000, size 99771
    Batch: 500000, size 99791
    Batch: 600000, size 99818
    Batch: 700000, size 99793
    Batch: 800000, size 99766
    Batch: 900000, size 99804
    Batch: 1000000, size 99790
    Batch: 1100000, size 99785
    Batch: 1200000, size 99799
    Batch: 1300000, size 99802
    Batch: 1400000, size 42979


```
len(data_set)
```




    1439986



```
# data_set.drop(columns='species_id', inplace=True)
print(f'Max Riqueza: {data_set[vocab].sum(axis=1).max()}')
data_set[data_set[vocab].sum(axis=1) == data_set[vocab].sum(axis=1).max()].T
```

    Max Riqueza: 39





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>93468</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bio_1</th>
      <td>20.8208</td>
    </tr>
    <tr>
      <th>bio_2</th>
      <td>11.5083</td>
    </tr>
    <tr>
      <th>bio_3</th>
      <td>39.9595</td>
    </tr>
    <tr>
      <th>bio_4</th>
      <td>651.618</td>
    </tr>
    <tr>
      <th>bio_5</th>
      <td>34.5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>16577</th>
      <td>0</td>
    </tr>
    <tr>
      <th>16801</th>
      <td>0</td>
    </tr>
    <tr>
      <th>16872</th>
      <td>0</td>
    </tr>
    <tr>
      <th>11917</th>
      <td>0</td>
    </tr>
    <tr>
      <th>17627</th>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3343 rows × 1 columns</p>
</div>



```
cont, cat = cont_cat_split(data_set, dep_var=vocab)
```

```
cat
```




    ['subset']



```
cont
```




    ['bio_1',
     'bio_2',
     'bio_3',
     'bio_4',
     'bio_5',
     'bio_6',
     'bio_7',
     'bio_8',
     'bio_9',
     'bio_10',
     'bio_11',
     'bio_12',
     'bio_13',
     'bio_14',
     'bio_15',
     'bio_16',
     'bio_17',
     'bio_18',
     'bio_19',
     'bdticm',
     'bldfie',
     'cecsol',
     'clyppt',
     'orcdrc',
     'phihox',
     'sltppt',
     'sndppt',
     'latitude',
     'longitude']



```
procs = [Categorify, FillMissing, Normalize]
```

```
y_names=vocab
```

```
cond = data_set.subset == 'train'

train_idx = np.where( cond)[0]
valid_idx = np.where(~cond)[0]

splits = (list(train_idx),list(valid_idx))
```

```
%%time 
to = TabularPandas(data_set,
                   cont_names=cont,
                   cat_names=None,
                   procs=procs,
                   y_names=y_names,
                   y_block=MultiCategoryBlock(encoded=True, vocab=vocab),
                   splits=splits)
```

    CPU times: user 1min 4s, sys: 26.8 s, total: 1min 30s
    Wall time: 1min 30s


```
len(to.train),len(to.valid)
```




    (1404515, 35471)



```
get_c(to)
```




    3313



```
dls = to.dataloaders(1024)
```

```
learn = tabular_learner(dls, [3000, 2000], metrics=accuracy_multi)
```

```
learn.model
```




    TabularModel(
      (embeds): ModuleList(
        (0): Embedding(3, 3)
        (1): Embedding(3, 3)
        (2): Embedding(3, 3)
        (3): Embedding(3, 3)
        (4): Embedding(3, 3)
        (5): Embedding(3, 3)
        (6): Embedding(3, 3)
        (7): Embedding(3, 3)
        (8): Embedding(3, 3)
        (9): Embedding(3, 3)
        (10): Embedding(3, 3)
        (11): Embedding(3, 3)
        (12): Embedding(3, 3)
        (13): Embedding(3, 3)
        (14): Embedding(3, 3)
        (15): Embedding(3, 3)
        (16): Embedding(3, 3)
        (17): Embedding(3, 3)
        (18): Embedding(3, 3)
        (19): Embedding(3, 3)
        (20): Embedding(3, 3)
        (21): Embedding(3, 3)
        (22): Embedding(3, 3)
        (23): Embedding(3, 3)
        (24): Embedding(3, 3)
        (25): Embedding(3, 3)
        (26): Embedding(3, 3)
      )
      (emb_drop): Dropout(p=0.0, inplace=False)
      (bn_cont): BatchNorm1d(29, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (layers): Sequential(
        (0): LinBnDrop(
          (0): Linear(in_features=110, out_features=3000, bias=False)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(3000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (1): LinBnDrop(
          (0): Linear(in_features=3000, out_features=2000, bias=False)
          (1): ReLU(inplace=True)
          (2): BatchNorm1d(2000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )
        (2): LinBnDrop(
          (0): Linear(in_features=2000, out_features=3313, bias=True)
        )
      )
    )



```
lr_min, lr_steep = learn.lr_find()
```






![png](GeoLifeCLEF-2021-tabular_files/output_49_1.png)


```
(lr_min, lr_steep)
```




    (0.13182567358016967, 0.02754228748381138)



```
learn.fit_one_cycle(15, lr_steep)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_multi</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.002262</td>
      <td>0.002264</td>
      <td>0.999696</td>
      <td>02:37</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.002221</td>
      <td>0.002224</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.002204</td>
      <td>0.002234</td>
      <td>0.999696</td>
      <td>02:37</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.002181</td>
      <td>0.002211</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.002169</td>
      <td>0.002206</td>
      <td>0.999697</td>
      <td>02:37</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.002155</td>
      <td>0.002205</td>
      <td>0.999697</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.002142</td>
      <td>0.002203</td>
      <td>0.999697</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.002126</td>
      <td>0.002193</td>
      <td>0.999697</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.002104</td>
      <td>0.002198</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.002081</td>
      <td>0.002184</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>10</td>
      <td>0.002050</td>
      <td>0.002178</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>11</td>
      <td>0.002018</td>
      <td>0.002169</td>
      <td>0.999697</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>12</td>
      <td>0.001985</td>
      <td>0.002162</td>
      <td>0.999697</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>13</td>
      <td>0.001959</td>
      <td>0.002162</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
    <tr>
      <td>14</td>
      <td>0.001944</td>
      <td>0.002160</td>
      <td>0.999696</td>
      <td>02:38</td>
    </tr>
  </tbody>
</table>


```
df_val = df.merge(df_env, on='observation_id')
```

```
df_val = df_val[df_val.subset == 'val']
y_val = df_val.species_id.astype(int)
```

```
dl_t = learn.dls.test_dl(df_val, bs=1024)
```

```
preds, targs = learn.get_preds(dl=dl_t)
```





```
spp_ids = array(y_names, dtype=int)
```

```
preds_ids = array([spp_ids[pred] for pred in predict_top_k_set(preds, 30)])
score_val = top_k_error_rate_from_sets(y_val, preds_ids)
print("Top-30 error rate: {:.1%}".format(score_val))
```

    Top-30 error rate: 77.8%


```
dl_t = learn.dls.test_dl(df_test.merge(df_env, on='observation_id'), bs=1024)
```

```
s_pred, _ = learn.get_preds(dl=dl_t)
```





```
preds_ids = array([spp_ids[pred] for pred in predict_top_30_set(s_pred)])
```

```
# Generate the submission file
generate_submission_file(SUBMISSION_PATH/"fastai_on_environmental_vectors.csv", df_test_obs_id_mapping["Id"], preds_ids)
```

```
!kaggle competitions submit -c geolifeclef-2021 -f {SUBMISSION_PATH/"fastai_on_environmental_vectors.csv"} -m "fastai submission"
```

    100%|██████████████████████████████████████| 5.76M/5.76M [00:03<00:00, 1.63MB/s]
    Successfully submitted to GeoLifeCLEF 2021 - LifeCLEF 2021 x FGVC8
