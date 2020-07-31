# Read in subject data and clean it


```python
# Import necessary packages 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
```


```python
# Read subjects' data.txt files into one list and concatenate the list into one DataFrame, data
data = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob('**/*data.txt'))], ignore_index=True)
```


```python
# Replace reaction time outliers (+/- 2 SD) with each participant's mean reaction times

# Add column of z-scores for individual's reaction times
data['zrt'] = (data.rt - data.groupby('id')['rt'].transform('mean')) / data.groupby('id')['rt'].transform('std')
# Replace +/- 2 SD with each participant's mean RTs
data.rt.where(abs(data.zrt) < 2, data.groupby('id')['rt'].transform('mean'), inplace=True, axis = 0)
```
