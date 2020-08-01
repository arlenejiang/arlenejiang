# Code Sample from NESC 3505: Assignment 2
### Looping Through Files

This code automates data analysis, reading data files and outputting mean reaction times for each subject. 

```python

# List of subject IDs, which correspond to file names
IDs = ['s10', 's12', 's13', 's14', 's15']
```



```python
for subject in IDs:
    file = pd.read_csv(subject + '.csv')
    print(subject)
    print('Mean reaction time in the congruent condition for ' + subject + ' is ' + str(file.loc[file['flankers'] == 'congruent', 'rt'].mean() * 1000) + ' ms')
    print('Mean reaction time in the incongruent condition for ' + subject + ' is ' + str(file.loc[file['flankers'] == 'incongruent', 'rt'].mean() * 1000) + ' ms')
```

    s10
    Mean reaction time in the congruent condition for s10 is 416.0478079416665 ms
    Mean reaction time in the incongruent condition for s10 is 439.95075269166665 ms
    s12
    Mean reaction time in the congruent condition for s12 is 415.1190512916668 ms
    Mean reaction time in the incongruent condition for s12 is 477.4034515217392 ms
    s13
    Mean reaction time in the congruent condition for s13 is 454.87979513333335 ms
    Mean reaction time in the incongruent condition for s13 is 497.66841203333297 ms
    s14
    Mean reaction time in the congruent condition for s14 is 470.89698339166677 ms
    Mean reaction time in the incongruent condition for s14 is 511.5541685999998 ms
    s15
    Mean reaction time in the congruent condition for s15 is 350.57277125217377 ms
    Mean reaction time in the incongruent condition for s15 is 380.82104304385956 ms

[⟵ Back](https://arlenejiang.github.io/arlenejiang/)
