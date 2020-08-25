# Code Sample from NESC 3505: Project 2
### Performing Statistical Tests

I collaborated on this code for a group project, where we analyzed EEG data for event related potentials if subjects read different types of sentences. The following code extracts relevant information from a DataFrame (test_1), calculates mean potentials across electrodes, and conducts a paired t-test between conditions (sentence types). This process is repeated for each pair of conditions (contrasts) and time windows of interest. 

```python
# Format output
report = "Time: {time}, Contrast: {contrast}; t({df})={t_val:.3f}, p={p:.3f}" 
print("\nTargeted Statistical Test Results:")
print('==================================')

# Loop through time windows
for time in time_windows:

    # Loop through contrasts
    for con in contrasts:
        
        # Slice DataFrame for relevant time/condition and average across electrodes
        A = test_1[(test_1['t_window'] == time) & (test_1['condition'] == contrasts[con][0])]['value']
        A_means = A.groupby(A.index//len(roi_labels['Mmid'])).mean()
        B = test_1[(test_1['t_window'] == time) & (test_1['condition'] == contrasts[con][1])]['value']
        B_means = B.groupby(B.index//len(roi_labels['Mmid'])).mean()

        # Conduct pairwise t-test
        test, p = stats.ttest_rel(A_means, B_means)

        # Display results
        format_dict = dict(time=time, contrast=con,  df=199, t_val=test, p=p)
    
        print(report.format(**format_dict))
    print()
```

    
    Targeted Statistical Test Results:
    ==================================
    Time: N400, Contrast: WPtn-Ctrl; t(199)=-22.656, p=0.000
    Time: N400, Contrast: RplusS-Ctrl; t(199)=-15.086, p=0.000
    Time: N400, Contrast: RnotS-Ctrl; t(199)=-19.271, p=0.000
    Time: N400, Contrast: RplusS-RnotS; t(199)=1.891, p=0.059
    
    Time: P600, Contrast: WPtn-Ctrl; t(199)=1.513, p=0.130
    Time: P600, Contrast: RplusS-Ctrl; t(199)=3.468, p=0.001
    Time: P600, Contrast: RnotS-Ctrl; t(199)=2.600, p=0.009
    Time: P600, Contrast: RplusS-RnotS; t(199)=1.109, p=0.267


[⟵ Back](https://arlenejiang.github.io/arlenejiang/)
