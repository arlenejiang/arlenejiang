```python
# Open 'spid...' folders, read in subjects' data.txt files into one list, concatenate list into one DataFrame; data
data = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob('**/*data.txt'))], ignore_index=True)
```
