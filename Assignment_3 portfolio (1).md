```python
# Plot histogram
df.rt.plot(kind = 'hist')

# Don't modify the code below here
# Add a solid line at the median and dashed lines at the 25th and 75th 
# percentiles (done for you)
plt.axvline(df['rt'].describe()['25%'], 0, 1, color='turquoise', linestyle='--')
plt.axvline(df['rt'].median(), 0, 1, color='cyan', linestyle='-')
plt.axvline(df['rt'].describe()['75%'], 0, 1, color='turquoise', linestyle='--')

# Rememebr to use plt.show() to see your plots (often they show anyway, but with some garbagy text at the top)
plt.show()
```




![png](Assignment_3%20portfolio_files/Assignment_3%20portfolio_0_0.png)




```python
# Plot the CDF
df.rt.plot(kind='hist', cumulative=True, density=True)

# Don't modify the code below here
# Add a solid line at the median and dashed lines at the 25th and 75th 
# percentiles (done for you)
plt.axvline(df['rt'].describe()['25%'], 0, 1, color='turquoise', linestyle='--')
plt.axvline(df['rt'].median(), 0, 1, color='cyan', linestyle='-')
plt.axvline(df['rt'].describe()['75%'], 0, 1, color='turquoise', linestyle='--')

plt.show()
```




![png](Assignment_3%20portfolio_files/Assignment_3%20portfolio_1_0.png)




```python
# Display boxplot of RTs by flanker and simon conditions
df.groupby(['flankers', 'simon']).plot(y='rt', kind='box')
```




    flankers     simon      
    congruent    congruent      AxesSubplot(0.125,0.125;0.775x0.755)
                 incongruent    AxesSubplot(0.125,0.125;0.775x0.755)
    incongruent  congruent      AxesSubplot(0.125,0.125;0.775x0.755)
                 incongruent    AxesSubplot(0.125,0.125;0.775x0.755)
    dtype: object






![png](Assignment_3%20portfolio_files/Assignment_3%20portfolio_2_1.png)






![png](Assignment_3%20portfolio_files/Assignment_3%20portfolio_2_2.png)






![png](Assignment_3%20portfolio_files/Assignment_3%20portfolio_2_3.png)






![png](Assignment_3%20portfolio_files/Assignment_3%20portfolio_2_4.png)


