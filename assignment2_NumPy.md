# Code Sample from NESC 3505: Assignment 2
### Manipulating NumPy Arrays

This sample showcases my ability to initalize and perform calculations with NumPy arrays. 

```python
# Import numpy
import numpy as np
```

```python
# Create a 2D NumPy array with reaction time and error data
full_data = np.array([rt, err])
#Convert reaction time from s to ms
full_data[0] = full_data[0]*1000
```

```python
# Compute mean reaction time
print("Mean RT = " + str(round(np.mean(full_data[0]), 1)) + " ms")
```

    Mean RT = 428.0 ms


```python
# Compute median reaction time
print("Median RT = " + str(round(np.median(full_data[0]), 1)) + " ms")
```

    Median RT = 410.2 ms


```python
# Compute standard deviation
print("Standard deviation = " + str(round(np.std(full_data[0]), 1)) + " ms")
```

    Standard deviation = 94.8 ms



