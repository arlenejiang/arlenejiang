# Code Sample from NESC 3505: Project 1
### Reading and Cleaning Data

Data for this project were obtained from an experiment examining reaction times under four different conditions (flanker and Simon effect). The following code reads 21 subjects' data from text files into a DataFrame. Reaction time data are then cleaned, replacing data associated with errors and that are outliers. 


```python
# Import necessary packages 
import pandas as pd
from glob import glob
```


```python
# Read subjects' data.txt files into one list and concatenate the list into one DataFrame, data
data = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob('spid**/*_data.txt'))], ignore_index=True)
```


```python
# Replace reaction times where subject made an error with NaN
data.loc[data['error']==True, 'rt'] = np.nan
```
```python
# Add column of z-scores for individual's reaction times
data['zrt'] = (data['rt'] - data.groupby('id')['rt'].transform('mean')) / data.groupby('id')['rt'].transform('std')
```
```python
# Replace +/- 2 SD with each participant's mean RTs
data.loc[((data['rt'].notna()) & abs(data['zrt'] > 2)), 'rt'] = data.groupby('id')['rt'].transform('mean')
```



[⟵ Back](https://arlenejiang.github.io/arlenejiang/)
