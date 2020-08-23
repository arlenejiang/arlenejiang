```python
import scipy.io as sio
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual
```


```python
cond_labels = ['CTRL', 'ADAPT'] 
contr_labels = [4, 8, 12, 16, 24, 32, 48, 64, 84, 100]
rep_labels = list(np.arange(1, 8))
num_reps = len(rep_labels)

time_labels = list(np.arange(4000))

stim_on_time = 2000
stim_off_time = stim_on_time + 1000

adapt_on_time = 0
adapt_off_time = adapt_on_time + 2000
```

### Plot Experimental Design

Some of the code below may come in handy later!

## Read in Matlab data file


```python
dat = sio.loadmat('crowder_single_unit.mat')
```

## Figure out the structure of the data

<font color='#0F4C81'>
<h2> 
    Q1
    </h2>
What is the type of `dat`? (answer using code output)
</font>


```python
# Display the type of dat
type(dat)
```




    dict



<font color='#0F4C81'>
<h2> 
    Q2
    </h2>
What is the length of `dat`? (answer using code output)
</font>


```python
# Display the length of dat
len(dat)
```




    4



In the Matlab file, the data are in 'SaveForAaron_May11_2020' (named after the original name that Dr. Crowder gave to the file when sharing it):


```python
dat.keys()
```




    dict_keys(['__header__', '__version__', '__globals__', 'SaveForAaron_May11_2020'])



<font color='#0F4C81'>
<h2> 
    Q3
    </h2>
What is the type of `'SaveForAaron_May11_2020'`? (answer using code output)
</font>


```python
# Display type of item in dat
type(dat['SaveForAaron_May11_2020'])
```




    numpy.ndarray



<font color='#0F4C81'>
<h2> 
    Q3b
    </h2>
What is the shape of `'SaveForAaron_May11_2020'`? (answer using code output)
</font>


```python
# Display shape of item in dat
dat['SaveForAaron_May11_2020'].shape
```




    (1, 23)



To see the what's stored for each neuron, we need to index the row (always 0, since there's only one row), and the appropriate column for that neuron. So for the first neuron:


```python
dat['SaveForAaron_May11_2020'][0,0]
```




    (array(['m1_6'], dtype='<U4'), array([[[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           ...,
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]]], dtype=uint8))



<font color='#0F4C81'>
<h2> 
    Q4
    </h2>
Modify the command in the previous code cell, to visualize the data from the last neuron in the set:
</font>


```python
# Display data for the last neuron in 'SaveForAaron_May11_2020'
dat['SaveForAaron_May11_2020'][-1,-1]
```




    (array(['m6_17b'], dtype='<U6'), array([[[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           ...,
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]],
    
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]]], dtype=uint8))



You can see that each of those entries is entirely contained in parentheses, indicating it's a tuple. So rather than the numpy `.shape()` method we need to use the `len()` function:


```python
len(dat['SaveForAaron_May11_2020'][0,0])
```




    2



The first entry in the tuple is the label of the neuron. Somewhat awkwardly, the label is actually buried inside a numpy array:


```python
dat['SaveForAaron_May11_2020'][0,0][0]
```




    array(['m1_6'], dtype='<U4')



So we have another level of indexing to grab the label:


```python
dat['SaveForAaron_May11_2020'][0,0][0][0]
```




    'm1_6'



The second entry in each neuron's tuple is the data:


```python
dat['SaveForAaron_May11_2020'][0,0][1].shape
```




    (4000, 8, 20)



## Convert to pandas DataFrame
### Create lists of condition labels

The next cell sets up labels for the rows of the DataFrame we're going to create. It's always important, in multi-dimensional data like this, to know what variables change "fastest" and which change "slowest". "Fastest means, from one row to the next, the value of that variable changes. In our case, we are going to organize the DataFrame so that all of the data for one neuron come before any data from the next neuron. So `neuron` changes slowest. `time` will change fastest, because we want all the time points of one trial (`repetition`), in sequence, before any time points of the next trial. `repetition` changes next-fastest, followed by `contrast` level, followed by `condition`.

Below this we use numpy's `.reshape()` method with the `newshape` argument set to `-1` (to reshape the 3D matrix into a 1D vector) and  `order='F'`. The latter tells the method to read / write the elements with the first index (dimension of the input data) changing fastest, and the last index changing slowest (if you're curious, the `F` is for Fortran, a language which uses this ordering). In our case, the first index is `time`. We know this because the shape of the input data is (4000, 8, 20), so time is the first dimension of the data. and it makes sense to list all the time points for one trial before moving on to the next. We do need to be aware of this so that we can set up the labels for each data point correctly. I've done this for you though, because getting it right is tricky, and if you get it wrong, all your results will be wrong (speaking as someone who learned the hard way...). 


```python
# Compute the total number of data points per neuron; we need to know how
#   many rows our DataFrame needs to have.
len_data = [(x * y * z) for x, y, z in [dat['SaveForAaron_May11_2020'][0][0][1].shape]][0]

# Set up vectors to label the columns in the pandas DataFrame
num_tp = dat['SaveForAaron_May11_2020'][0,0][1].shape[0]
time_labels = list(np.arange(num_tp))
times = time_labels * (len_data//len(time_labels))

num_condcontr = dat['SaveForAaron_May11_2020'][0,0][1].shape[2]
num_cond = len(cond_labels)
num_contr = len(contr_labels)

num_reps = dat['SaveForAaron_May11_2020'][0,0][1].shape[1]
rep_labels = list(np.arange(1, num_reps+1))
reps = np.tile(np.repeat(rep_labels, num_tp), num_cond * num_contr)

# Since condition and contrast are actually separate variables, we'll
#  break them out here.
contrs = np.tile(np.repeat(contr_labels, num_tp * num_reps), num_cond)
conditions = np.repeat(cond_labels, num_tp * num_reps * num_contr)

neuron_labels = ['m1_6', 'm1_12', 'm3_4', 'm3_11', 'm6_3a2', 'm6_11']
num_neurons = len(neuron_labels)
```

Below is the code that creates the DataFrame. 

Building a DataFrame with 2.5 m rows takes some time, so be patient while this runs:


```python
df_list = []

for n in range(num_neurons):
    neurons = np.repeat(neuron_labels[n], len_data)
    df_tmp = pd.DataFrame(zip(neurons, times, reps, contrs, conditions,
                              dat['SaveForAaron_May11_2020'][0][n][1].reshape(-1, order='F')[None][0]), 
                      columns = ['neuron', 'time', 'repetition', 'contrast', 'condition', 'spike']
                     )
    df_list.append(df_tmp)
    
df = pd.concat(df_list)

# Clear things from memory so CoCalc doesn't run out
del df_tmp
del dat
```

<font color='#0F4C81'>
<h2> 
    Q5
    </h2>
    
Show the DataFrame, `df`:
</font>


```python
# Display the DataFrame
df
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
      <th>neuron</th>
      <th>time</th>
      <th>repetition</th>
      <th>contrast</th>
      <th>condition</th>
      <th>spike</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>m1_6</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>CTRL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>m1_6</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>CTRL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>m1_6</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>CTRL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>m1_6</td>
      <td>3</td>
      <td>1</td>
      <td>4</td>
      <td>CTRL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>m1_6</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>CTRL</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>639995</th>
      <td>m6_11</td>
      <td>3995</td>
      <td>8</td>
      <td>100</td>
      <td>ADAPT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>639996</th>
      <td>m6_11</td>
      <td>3996</td>
      <td>8</td>
      <td>100</td>
      <td>ADAPT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>639997</th>
      <td>m6_11</td>
      <td>3997</td>
      <td>8</td>
      <td>100</td>
      <td>ADAPT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>639998</th>
      <td>m6_11</td>
      <td>3998</td>
      <td>8</td>
      <td>100</td>
      <td>ADAPT</td>
      <td>0</td>
    </tr>
    <tr>
      <th>639999</th>
      <td>m6_11</td>
      <td>3999</td>
      <td>8</td>
      <td>100</td>
      <td>ADAPT</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3840000 rows × 6 columns</p>
</div>



<font color='#0F4C81'>
<h2> 
    Q14
    </h2>

### Heat maps
    
Plot heat maps showing the PTSHs, with colour indicating the spike count. The great thing about heat maps is how condensed they are. We don't really need to use an interactive plot, because one plot includes all contrast levels. So plot this with two columns (CTRL, ADAPT) and 4 rows (each neuron).
    
For each heat map, put time on the *x* axis, contrast on the *y* axis, and colour indicating the number of spikes per bin. 

For heat maps you need to compute the histograms for all contrast levels, and store these temporarily in an object such as a list dictionary, or numpy array.
</font>



```python
# Define bins for histograms
hist_bin_width = 50 
time_bins = np.arange(0, max(time_labels), hist_bin_width)
# The following code plots all neurons in neuron_labels
# *** To simply plot the first 4 neurons, I would use the slice neuron_labels[0:4] in place of neuron_labels
# Intialize figure
fig = plt.figure(figsize=[16, 8])

# Name figure
plt.suptitle('Heat maps of mean spiking to stimuli of different contrast levels', fontsize=16)

# Initialize subplot counter and max spikes for color bar
subplot_counter = 1
max_spikes = 0

# Loop through neurons or neuron_labels[0:4] see ** in line 2
for neuron in neuron_labels:
    # Slice relevant section of DataFrame
    neu_dat = df[(df['neuron'] == neuron)]

    # Loop through conditions
    for cond in cond_labels:
        # Slice relevant section of DataFrame
        tmp_dat = neu_dat[(neu_dat['condition'] == cond)]
    
        # Initialize dictionary for PSTHs in the condition
        psth_temp = {}
        
        # Add subplot for each condition
        ax = fig.add_subplot(6, 2, subplot_counter)

        # Loop through contrasts
        for contr in contr_labels:
            # Slice relevant section of DataFrame
            spikeTimes = tmp_dat[(tmp_dat['contrast'] == contr) & (tmp_dat['spike'] == 1)]['time']
            
            # Calculate PSTHs
            nOut, bins = np.histogram(spikeTimes, bins=time_bins)
            psth_temp[contr] = nOut/(num_reps*hist_bin_width)

        # Plot heat map for one neuron's condition
        hmap = ax.imshow([psth_temp[i] for i in sorted(psth_temp.keys())], extent = [min(time_labels), max(time_labels)+1, 10, 0], 
                         cmap='viridis', interpolation='bilinear', aspect='auto')

        # Find heat map with maximum spikes for color bar
        temp_max_spikes = max([max(p) for p in [psth_temp[i] for i in sorted(psth_temp.keys())]])
        if temp_max_spikes > max_spikes:
            max_map = hmap
            max_spikes = temp_max_spikes

        # Format plot

        # Title
        if subplot_counter == 1:
            plt.title('CTRL condition')
        if subplot_counter == 2:
            plt.title('ADAPT condition')

        # X-Axis
        if subplot_counter > 10:
            plt.xlabel('Time (ms)')
        else:
            plt.xticks([])
        
        #y-axis
        if subplot_counter % 2 == 1:
            plt.yticks([0.5, 2.5, 4.5, 6.5, 8.5], contr_labels[::2])
            plt.ylabel(neuron + '\nContrast level', )
        else:
            plt.yticks([])

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        # Increase subplot counter
        subplot_counter += 1

# Set y-axis label
fig.subplots_adjust(left = 0.07)
fig.text(0.01, 0.5, 'Neurons', fontsize=14, ha='center', va='center', rotation='vertical')

# Plot colour bar from heat map with maximum mean spiking per bin
fig.subplots_adjust(right=0.95)
cb = fig.colorbar(max_map, cax=fig.add_axes([0.96, 0.072, 0.025, 0.805]))
cb.ax.set_ylabel('Mean spikes per time bin', fontsize=14)

# Display plot
plt.show()
```




![png](Portfolio%20-%20Assignment_4_files/Portfolio%20-%20Assignment_4_36_0.png)



<font color='#0F4C81'>
<h2> 
    Q16
    </h2>
    
## Contrast Response Functions

Plot stimulus contrast (*x*) against the mean spike rate (*y*) during the stimulus "on" period (2000–3000 ms) to produce a contrast response function (CRF), which is a roughly sigmoid-shaped function.

Plot the CRF for the two conditions on the same plot, with one subplot per neuron. 
    </font>


```python
# Initialize figure
fig = plt.figure(figsize=([12,12]))

# Name figure
plt.suptitle('Contrast response functions for each neuron', fontsize=16)

# Slice relevant section of DataFrame
stim_df = df[(df['time'] >= 2000) & (df['time'] < 3000)]

# Initialize subplot counter
subplot_counter = 1

# Loop through neurons
for neuron in neuron_labels:
    
    # Examine relevant section of DataFrame
    neu_dat = stim_df[stim_df['neuron'] == neuron]
    
    # Add subplot for each neuron
    ax = fig.add_subplot(3, 2, subplot_counter)
    
    # Loop through conditions
    for condition in cond_labels:
        
        # Set plotting colour depending on condition
        if condition == 'CTRL':
            cond_color = 'blue'
        else:
            cond_color = 'red'
        
        # Calculate mean spike rates for each contrast level
        mean_spike_rate = [neu_dat[(neu_dat['condition'] == condition) & (neu_dat['contrast'] == contrast)]['spike'].mean()
                           for contrast in contr_labels]
        # Plot spike rates by contrast labels
        line = plt.plot(contr_labels, mean_spike_rate, color=cond_color, marker='o', label=condition)
        
    
    # Format plot
    plt.title(neuron)
    if subplot_counter == 2:
        plt.legend()

    # X-Axis
    plt.xlim([0, contr_labels[-1]+4])
    if subplot_counter > 4:
        plt.xlabel('Stimulus contrast level')
    else:
        plt.xticks([])

    # Y-Axis
    plt.ylim([0, 0.04])
    if subplot_counter % 2 == 1:
        plt.ylabel('Mean spike rate (spikes per ms)')
    else:
        plt.yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Increment counter
    subplot_counter+=1

# Display plot
plt.show()
```




![png](Portfolio%20-%20Assignment_4_files/Portfolio%20-%20Assignment_4_38_0.png)




# Q17 
# Neural Latency (latency to first spike)
You can measure neural latency in the control condition, and latency is expected to decrease with increasing contrast.

Using a format similar to the CRFs, make a plot of latency to first spike (computed as the mean across repetitions) on the y axis, against contrast on the x axis, using one subplot per neuron (with both conditions in one subplot).


```python
# Initialize figure
fig = plt.figure(figsize=([12,12]))

# Name figure
plt.suptitle('Neural latency for each neuron', fontsize=14)

# Initialize subplot counter
subplot_counter = 1

# Slice relevant section of DataFrame
stim_df = df[(df['time'] >= 2000) & (df['time'] < 3000)]

# Loop through neurons
for neuron in neuron_labels:

    # Add subplot for each neuron
    ax = fig.add_subplot(3, 2, subplot_counter)

    # Slice relevant section of DataFrame
    neu_dat = stim_df[(stim_df['neuron'] == neuron)]
    
    # Loop through conditions
    for condition in cond_labels:

        # Set plotting colour depending on condition
        if condition == 'CTRL':
            cond_color = 'blue'
        else:
            cond_color = 'red'

        # Examine relevant section of DataFrame
        temp = neu_dat[(neu_dat['condition'] == condition)]
        
        # Calculate mean latency
        mean_latency = [(np.nansum([temp[(temp['spike'] == 1) 
                                            & (temp['repetition'] == rep) 
                                            & (temp['contrast'] == contrast)]['time'].min() - 2000
                                            for rep in rep_labels])/num_reps) for contrast in contr_labels]

        # Plot spike rates by contrast labels
        plt.plot(contr_labels, mean_latency, color=cond_color, marker='o', label=condition)

    # Format plot
    plt.title(neuron)
    if subplot_counter == 2:
        plt.legend()
    
    # X-Axis
    plt.xlim([0, contr_labels[-1]+4])
    if subplot_counter > 4:
        plt.xlabel('Stimulus contrast level')
    else:
        plt.xticks([])

    # Y-Axis
    plt.ylim([0, 450])
    if subplot_counter % 2 == 1:
        plt.ylabel('Latency to first spike (ms)')
    else:
        plt.yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # Increment counter
    subplot_counter+=1

# Display plot
plt.show()
```




![png](Portfolio%20-%20Assignment_4_files/Portfolio%20-%20Assignment_4_40_0.png)



# THE END
