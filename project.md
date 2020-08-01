# Code Sample from NESC 3505: Project 1
### Reading and Cleaning Data

Data for this project were obtained from an experiment examining reaction times under four different conditions (flanker and Simon effect). The following code reads 21 subjects' data from text files into a DataFrame. Reaction time outliers for each subject are then removed using the groupby function. 


```python
# Import necessary packages 
import pandas as pd
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

[⟵ Back](https://arlenejiang.github.io/arlenejiang/)
