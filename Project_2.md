# Project 2
<font color='#0F4C81'>
<h2>NESC 3505</h2>
    </font>

## Group-Level ERP Visualization & Anaysis
# MSA Experiment

These data are from an honours project designed and run by Yasmin Beydoun in 2015-16, in collaboration with Drs. Kaitlyn Tagarelli and Aaron Newman, Neurocognitive Imaging Lab, Department of Psychology & Neuroscience, Dalhousie University. 

In this study, native Arabic language speakers read a set of sentences written in Modern Standard Arabic (MSA). While different Arabic-speaking countries and groups have distinct dialects, MSA is the standardized written form of the language used across the Arab world. Arabic languages, including MSA, have a relatively unusual *morphology* — rules specifying how words are formed. While in many languages (including English) words can be formed by adding prefixes to the start of a word or suffixes to the end of a word, Arabic uses *infixes*. That is, the "root" of words is a set of consonants, and grammatical rules specify how vowels are inserted (infixed) in between the consonants to form different words, with different meanings and/or grammatical properties. For example, the root *KTB* (note we are translating Arabic letters into English letters with similar sounds, for simplicity of explanation) is associated with the concept of "writing", and can form words such as *kitab* (book) and *kataab* (writing). The consonant root can be referred to as the *semantic* root, because it carries the core semantic (meaning) properties of most words derived from it (as in the KTB example) while the vocalic (vowel) morphemes are termed *syntactic* because particular vocalic patterns carry common syntactic (grammatical) patterns across words. For example, the morpheme *a_a_a* marks the past tense of a verb, so wrote would be *kataba*; the morpheme *_aa_i_* marks the active participle (as in *kaatib*, “writer”). 

## Experimental Design
Participants read MSA sentences while EEG data were recorded. The sentences were either well-formed according to the rules of Arabic, or ended in words with particular violations of Arabic word formation rules. This use of violations is common in ERP studies of sentence processing, and different types of violations elicit distinct ERP effects. Specifically, violations of semantics, (e.g., *I take my coffee with milk and dog.*), elicit an **N400** relative to control sentences (*I take my coffee with milk and sugar*). The N400 is an enhanced negative potential for violations, largest over electrodes on the top (midline-centre) of the head, that tends to occur between 300–600 ms after word onset. In contrast, syntactic violations elicit a **P600**, which manifests as a positivity largest over midline central-to-posterior electrodes, between 600–900 ms. We designed teh violations in this experiment with teh expectation that  

The sentences in this experiment were designed to fit into four conditions, including well-formed (control) sentences, and three types of violations of Arabic word formation: morpho-semantic, morpho-syntactic (semantically related), and morpho-syntactic (semantically unrelated). Each sentence was presented in all four conditions; in other words, first a well-formed version of a sentence was produced, and then version of that sentences containing each of the three violation types were produced from it. There were 40 sentences in each condition, for a total of 160 sentences. Because eye movements produce ERP artifacts, the sentences were presented one word at a time, on a computer screen, with a new word presented every 700 ms. After the end of each sentence, participants were asked to rate how "good" or "bad" the sentence was, on a 5 point Likert scale. 


| Condition | Label | English translation | Arabic |
| --- | --- | --- | --- |
| Correct sentence | `Ctrl` | The boy was standing next to the classroom **entrance** (Madxal) | الْمَدْخَلِ بِجَانِبِ لصَّبِيُّا وَقَفَ |
| Semantic violation | `WPtn` | The boy was standing next to the classroom **the killing** (MaQtal) | الْمَقْتَلِ بِجَانِبِ لصَّبِيُّا وَقَفَ |
| Morpho-syntactic violation (semantically related) | `RplusS` | The boy was standing next to the classroom **entering** (Duxul) | الدُّخُولِ بِجَانِبِ لصَّبِيُّا وَقَفَ |
| Morpho-syntactic violation (semantically unrelated) | `RnotS` | The boy was standing next to the classroom **participation** (MuDa:Xalatun) | الْمُداخَلَةِ بِجَانِبِ لصَّبِيُّا وَقَفَ |


## About the Data

The data here are from 18 participants. For each participant, continuous EEG data were recorded from 64 scalp electrodes, as well as two EOG (electroculogram) channels to monitor for eye artifacts. 
Each data set was preprocessed using MNE-Python, including (in order) bandpass filtering (0.1 – 30 Hz), removal of any excessively noisy channels or trials, artifact correction using ICA, and interpolation of data for any removed channels. The data from all trials in each condition were then averaged together. 

The result, provided to you, is an MNE Evoked file for each participant, containing the averaged ERP waveforms for each condition, and each electrode. 




## Hypotheses

We predicted the following ERP effects for the contrasts between pairs of conditions:

| Contrast | ERP Effects | 
| --- | --- | 
| WPtn - Ctrl | N400 |
| RplusS - Ctrl | N400 + P600 |
| RnotS - Ctrl | P600 | 
| RplusS - RnotS | N400 |

For the purposes of this analysis, you will use 400–600 ms as the time window for the N400, and 700–900 ms for the P600. For both, you will focus on a "region of interest" (ROI) comprised of electrodes around the top of the head, for both the N400 and P600. THese parameters are already set for you in the code below.

## Your Tasks

Test the hypotheses above using
- topographic scalp maps in time windows of interest
- waveform plots at electrodes of interest
- Seaborn plots of mean amplitudes for conditions/contrasts within ROIs
- t-tests at selected electrodes
- mass univariate analyses

The instructions below will guide you through. There are a number of distinct sections to the assignment that can be worked on semi-independently, although all depend on first getting the data imported and organized. The MNE commands you will need to run are all provided for you in the instructions. You will want to consult the [MNE API](https://mne.tools/stable/python_reference.html) for instructions on how to use the different MNE commands. Beyond the MNE commands provided, you should be able to complete all the tasks using your existing knowledge of Python, including lists, dictionaries, np arrays, pandas DataFrames, and Seaborn plots.

## Initialization


```python
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats

import mne
import mne.stats
from mne.viz import plot_compare_evokeds
from mne.channels import find_ch_connectivity, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test
```

## Read in data

Data are saved in files for each subject, continaing a set of four MNE `Evoked` instances. Each evoked object in the file for a subject represents the data for a particular condition, averaged over trials. All preprocessing has been performed already. 


```python
data_path = 'evoked/'
```

Read in the data to a dictionary called `evoked`, with the dictionary keys being the ID codes for each subject (e.g., `MSA_01`, as the files are named). Use `mne.read_evokeds`. Besides specifying the file name, the only argument you should pass to `mne.read_evokeds` is `baseline=(None, 0)`


```python
subjects= ['MSA_01', 'MSA_02', 'MSA_03', 'MSA_04', 'MSA_05', 'MSA_07', 'MSA_08', 'MSA_09', 'MSA_10', 'MSA_11', 'MSA_12', 'MSA_13', 'MSA_14', 'MSA_15', 'MSA_16', 'MSA_17', 'MSA_18', 'MSA_19']
evoked = {}
counter=0
for file in subjects:
    in_file= data_path + file + '-ave.fif'
    ev= {subjects[counter] : mne.read_evokeds(in_file, baseline=(None,0))}
    evoked.update(ev)
    counter += 1
```

    Reading evoked/MSA_01-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 30 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_02-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 36 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_03-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 40 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_04-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 34 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 36 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_05-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 38 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 39 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 38 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 38 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_07-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 39 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 39 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_08-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 19 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 18 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 18 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 19 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_09-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 34 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_10-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 39 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 38 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_11-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 38 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 38 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 36 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 39 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_12-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 34 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 31 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 34 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_13-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 37 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_14-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 13 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 16 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 19 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 17 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_15-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 16 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 18 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 16 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 17 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_16-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 29 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 28 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 28 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_17-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 24 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 31 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 29 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_18-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 36 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 36 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 42 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 39 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
    Reading evoked/MSA_19-ave.fif ...
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (Ctrl)
            0 CTF compensation matrices available
            nave = 33 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RnotS)
            0 CTF compensation matrices available
            nave = 35 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (RplusS)
            0 CTF compensation matrices available
            nave = 36 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)
        Found the data of interest:
            t =    -199.22 ...    1000.00 ms (WPtn)
            0 CTF compensation matrices available
            nave = 31 - aspect type = 100
    No projector specified for this dataset. Please consider the method self.add_proj.
    Applying baseline correction (mode: mean)



Show the `evoked` dictionary:


```python
evoked
```




    {'MSA_01': [<Evoked  |  'Ctrl' (average, N=30), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_02': [<Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_03': [<Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=40), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_04': [<Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_05': [<Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_07': [<Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_08': [<Evoked  |  'Ctrl' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_09': [<Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_10': [<Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_11': [<Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_12': [<Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_13': [<Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_14': [<Evoked  |  'Ctrl' (average, N=13), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_15': [<Evoked  |  'Ctrl' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_16': [<Evoked  |  'Ctrl' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_17': [<Evoked  |  'Ctrl' (average, N=24), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_18': [<Evoked  |  'Ctrl' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=42), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_19': [<Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>]}



Show the values for the first subject in the `evoked` dictionary:


```python
print(list(evoked.values())[0])
#To find the keys (subject) just change .values with .keys
```

    [<Evoked  |  'Ctrl' (average, N=30), [-0.19922, 1] sec, 64 ch, ~477 kB>, <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>, <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>, <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>]


What is the `type` of each entry in the `evoked` dictionary?


```python
for x in subjects:
    print(type(evoked[x]))
```

    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>
    <class 'list'>



```python
evoked
```




    {'MSA_01': [<Evoked  |  'Ctrl' (average, N=30), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_02': [<Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_03': [<Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=40), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_04': [<Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_05': [<Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_07': [<Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_08': [<Evoked  |  'Ctrl' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_09': [<Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_10': [<Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_11': [<Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_12': [<Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_13': [<Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_14': [<Evoked  |  'Ctrl' (average, N=13), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_15': [<Evoked  |  'Ctrl' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_16': [<Evoked  |  'Ctrl' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_17': [<Evoked  |  'Ctrl' (average, N=24), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_18': [<Evoked  |  'Ctrl' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=42), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'MSA_19': [<Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>]}



Show how you would select the first item in the first dictionary entry:


```python
print(list(evoked.values())[0][0])
#can also use the key like: print(evoked["MSA_01"][0])
```

    <Evoked  |  'Ctrl' (average, N=30), [-0.19922, 1] sec, 64 ch, ~477 kB>


What is the `type` of the first item in the first dictionary entry? (Which will also be the type of every other entry)


```python
print(type(evoked["MSA_01"][0]))
#Or print(type(list(evoked.values())[0][0]))
```

    <class 'mne.evoked.Evoked'>


Conveniently (/by design), the conditions are listed in the same order for every subject, which makes it easy to iterate over subjects to access the data for a specific condition.

## Experimental conditions and contrasts

The `.comment` attributed of an `Evoked` object stores the name of the condition. Demonstrate how you would display this for the first condition of the first subject:


```python
evoked["MSA_01"][0].comment
```




    'Ctrl'



This makes it easy to make a list of conditions, which we can then iterate over as necessary. Create a list called `conditions`, using list comprehension, that contains the condition labels.


```python
conditions = [ ]
for comment in range(len(evoked["MSA_01"])):
    conditions.append(evoked["MSA_01"][comment].comment)
print(conditions)
```

    ['Ctrl', 'RnotS', 'RplusS', 'WPtn']


## Grand Averages
In ERP-speak, "grand average" is an average across subjects, for each condition. Generate grand averages in the cell below, saving the results in a dictionary called `gavg` that is keyed by condition label.

The function to create a grand average is `mne.combine_evoked()` include the argument `weights='equal'`

<font color='green'>
    <b>HINT:</b> don't forget about the `enumerate()` function...
</font>


```python
# Initialize the dictionary, a list to hold values, and a counter to indicate condition. 
gavg = {}

# Loop over conditions and use them as keys in the gavg dictionary
for i, cond in enumerate(conditions):
    condValues = []
    # Loop over every subject and add the value to the condValues list. 
    for subject in range(len(subjects)):
        condValues.append(list(evoked.values())[subject][i])
    # After we loop through every subject for a given condition and save the values in a list, send that list to the mne.combine_evoked method.
    newAvg = {cond : mne.combine_evoked(condValues, weights = 'equal')}
    # Add it to the dictionary, reset the list, and update the condition counter. 
    gavg.update(newAvg)
gavg
```




    {'Ctrl': <Evoked  |  '1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl' (average, N=1.559710309445706), [-0.19922, 1] sec, 64 ch, ~477 kB>,
     'RnotS': <Evoked  |  '1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS + 1.000 * RnotS' (average, N=1.6609980884756734), [-0.19922, 1] sec, 64 ch, ~477 kB>,
     'RplusS': <Evoked  |  '1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS + 1.000 * RplusS' (average, N=1.68250703361808), [-0.19922, 1] sec, 64 ch, ~477 kB>,
     'WPtn': <Evoked  |  '1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn + 1.000 * WPtn' (average, N=1.6736339390134967), [-0.19922, 1] sec, 64 ch, ~477 kB>}




```python
type(gavg)
```




    dict



Show the keys of `gavg`:


```python
gavg.keys()
```




    dict_keys(['Ctrl', 'RnotS', 'RplusS', 'WPtn'])



Show the value for the Control condition in the `gavg` dictionary:


```python
gavg['Ctrl']
```




    <Evoked  |  '1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl + 1.000 * Ctrl' (average, N=1.559710309445706), [-0.19922, 1] sec, 64 ch, ~477 kB>



Create an alternate version of `evoked`. Whereas `evoked` is keyed by subject, create `evoked_by_cond` such that is keyed by *condition*, with the dictionary values for each condition being a list of the data from that condition, for each subject. Re-organizing the evoked data this way is useful for some plots below, notably waveform plots with confidence intervals.


```python
print(conditions)
```

    ['Ctrl', 'RnotS', 'RplusS', 'WPtn']



```python
evoked_by_cond = {}
condValues2 = []
condCounter2 = 0

#Loop over conditions and use them as keys in the gavg dictionary
for cond2 in conditions:
    #Loop over every subject and add the value to the condValues list. 
    for subject2 in range(len(subjects)):
        condValues2.append(list(evoked.values())[subject2][condCounter2])
    #After we loop through every subject for a given condition and save the values in a list, send that list to the mne.combine_evoked method.
    newAvg2 = {cond2 : condValues2}
    #Add it to the dictionary, reset the list, and update the condition counter. 
    evoked_by_cond.update(newAvg2)
    condValues2 = []
    condCounter2 += 1
    
evoked_by_cond
```




    {'Ctrl': [<Evoked  |  'Ctrl' (average, N=30), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=13), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=24), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'RnotS': [<Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RnotS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'RplusS': [<Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=40), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=18), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=42), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'RplusS' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     'WPtn': [<Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>]}



## Difference waves

Key to testing our hypotheses are contrasts between conditions. The dictionary below specifies the contrasts of interest.


```python
contrasts = {'WPtn-Ctrl':['WPtn', 'Ctrl'],
             'RplusS-Ctrl':['RplusS', 'Ctrl'],
             'RnotS-Ctrl':['RnotS', 'Ctrl'],
             'RplusS-RnotS':['RplusS', 'RnotS']
            }
```

Create a dictionary called `gavg_diff` that contains the *difference waves* for each contrast. Difference waves are the differences between the two conditions, for every time point in the grand average. Use `mne.combine_evoked()`, with the first argument being a list of the two conditions you want to compare (with a minus sign preceding the second item in the list, so that it is subtracted from the first), and the second argument being `weights='equal'`. 


```python
iff1= mne.combine_evoked([gavg['WPtn'],-gavg['Ctrl']], weights='equal')
diff2= mne.combine_evoked([gavg['RplusS'],-gavg['Ctrl']], weights='equal')
diff3= mne.combine_evoked([gavg['RnotS'],-gavg['Ctrl']], weights='equal')
diff4= mne.combine_evoked([gavg['RplusS'],-gavg['RnotS']], weights='equal')

gavg_diff = {'WPtn-Ctrl': diff1, 'RplusS-Ctrl': diff2, 'RnotS-Ctrl': diff3, 'RplusS-RnotS': diff4}
```

## Topographic Maps

Topographic maps (topomaps) are plots of the scalp, as seen from above, that show the distribution of voltage values averaged over a certain time range as colours. 

The example below will plot the topomap for the control condition, averaged over a time window of 50 ms centred on 500 ms. Note that MNE data represents time in seconds, not milliseconds.

Another weird thing is that MNE plot commands tend to generate two copies of the plot. You can suppress this by putting a semicolon at the end of the plot command, as in the example below.


```python
gavg['Ctrl'].plot_topomap(0.500, 
                          average=.050);
```




![png](Project_2_files/Project_2_44_0.png)



In the cell below, write a loop that plots topomaps for each condition over a range of 100 ms intervals from 0-999 ms, each one averaging over a 50 ms window. In other words, each row should be a set of 10 topomaps at differnt time points, for a given condition. 

Use the `Evoked.plot_topomap` API to set the following options: no sensors, no contours, viridis colourmap, title = the condition label.


```python
times= np.arange(0, 0.999, 0.1)


#Order of conditions from top to bottom: "Ctrl", "RnotS", "RplusS", "WPtn"
for x in gavg:
        gavg[x].plot_topomap(times, average=0.050, sensors=False, contours=0, cmap='viridis', title= x);
```




![png](Project_2_files/Project_2_46_0.png)






![png](Project_2_files/Project_2_46_1.png)






![png](Project_2_files/Project_2_46_2.png)






![png](Project_2_files/Project_2_46_3.png)



## A priori time windows of interest

Based on hypotheses, define start & end of time windows to compute mean amplitudes over for each predicted ERP component. The dictionary below defines the time window labels and start/end times corresponding to our two predicted ERP components:


```python
time_windows = {'N400':(0.400, 0.600),
                'P600':(0.700, 0.900)
               }
```

Write a loop (hint: one loop nested inside another) that cycles through conditions, and for each one plots the topomaps averaged over each component's time window. The title of each plot show make it clear what condition and component it is.  


```python
for x in gavg:
    for time in time_windows:
        gavg[x].plot_topomap(time_windows[time], average=.2, contours=0, cmap='viridis', sensors=False, title= x + ' ' + str(time))
        
       
```




![png](Project_2_files/Project_2_50_0.png)






![png](Project_2_files/Project_2_50_1.png)






![png](Project_2_files/Project_2_50_2.png)






![png](Project_2_files/Project_2_50_3.png)






![png](Project_2_files/Project_2_50_4.png)






![png](Project_2_files/Project_2_50_5.png)






![png](Project_2_files/Project_2_50_6.png)






![png](Project_2_files/Project_2_50_7.png)



### Contrast topomaps

Plot the topomaps on 100 ms time intervals (each averaged over 50 ms), as you did above. This time, however, do this for the contrasts (using `gavg_diff`) rather than the individual conditions.


```python
times= np.arange(0, 0.999, 0.1)


for x in gavg_diff:
        gavg_diff[x].plot_topomap(times, average=0.050, sensors=False, contours=0, cmap='viridis', title= x);
```




![png](Project_2_files/Project_2_52_0.png)






![png](Project_2_files/Project_2_52_1.png)






![png](Project_2_files/Project_2_52_2.png)






![png](Project_2_files/Project_2_52_3.png)



### Difference topomaps for ERP component windows

Write a loop (hint: one loop nested inside another) that cycles through contrasts, and for each one plots the topomaps averaged over each component's time window. The title of each plot show make it clear what condition and component it is.  

Use the parameters `vmin=-20` and `vmax=20` so that all of the maps are on the same scale.


```python
for x in gavg_diff:
    for time in time_windows.keys():
        gavg_diff[x].plot_topomap(time_windows[time], average=.2, sensors=False, contours=0, cmap='viridis', vmin=-20, vmax=20, title= x + ' ' + str(time));
```




![png](Project_2_files/Project_2_54_0.png)






![png](Project_2_files/Project_2_54_1.png)






![png](Project_2_files/Project_2_54_2.png)






![png](Project_2_files/Project_2_54_3.png)






![png](Project_2_files/Project_2_54_4.png)






![png](Project_2_files/Project_2_54_5.png)






![png](Project_2_files/Project_2_54_6.png)






![png](Project_2_files/Project_2_54_7.png)



---
## Waveform plots

## ROIs
Here we define clusters of electrodes to group together for analysis. 

For reference, the locations of each channel on the head are shown below.

![](channel_map.png)


```python
roi_labels = {'Lant':['AF7', 'AF3', 'F7', 'F5', 'F3','FT9', 'FT7', 'FC5', 'FC3'],
              'Mant':['Fp1', 'Fp2', 'F1', 'Fz', 'F2', 'FC1', 'FC2'],
              'Rant':['AF4', 'AF8', 'F4', 'F6', 'F8','FC4', 'FC6', 'FT8', 'FT10'],
              'Lmid':['T7', 'C5', 'C3', 'TP7', 'CP5', 'CP3','P7', 'P5', 'P3'],
              'Mmid':['C1', 'Cz', 'C2', 'CP1', 'CPz', 'CP2','P1', 'Pz', 'P2'],
              'Rmid':['C4', 'C6', 'T8', 'CP4', 'CP6', 'TP8', 'P4', 'P6', 'P8'],
              'Mpost':['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'PO9', 'PO10']
             }
```

### Identify channel indices for ROIs

This step is done for you; it's necessary because many MNE functions identify channels by their index in an np array, rather than by their labels. The code below finds the channel indices for each label in each ROI.


```python
rois = {}

for region in roi_labels.keys():
    rois[region] = [gavg[list(gavg)[0]].ch_names.index(chan) for chan in roi_labels[region]]
    
rois
```




    {'Lant': [32, 33, 2, 36, 3, 40, 41, 7, 42],
     'Mant': [0, 1, 37, 4, 38, 8, 9],
     'Rant': [34, 35, 5, 39, 6, 43, 10, 44, 45],
     'Lmid': [11, 46, 12, 50, 17, 51, 22, 55, 23],
     'Mmid': [47, 13, 48, 18, 52, 19, 56, 24, 57],
     'Rmid': [14, 49, 15, 53, 20, 54, 25, 58, 26],
     'Mpost': [59, 60, 61, 62, 63, 28, 29, 30, 27, 31]}



### Visualize waveforms

This example will generate an ERP waveform plot averaged over all electrodes for the `Mmid` ROI, with different line styles for each condition in `gavg`:


```python
mne.viz.plot_compare_evokeds(gavg, 
                             picks=rois['Mmid'],
                             combine='mean',
                             linestyles={conditions[0]:'-', conditions[1]:'--', conditions[2]:':', conditions[3]:'-.'},
                             show_sensors=4,
                             );
```

    More than 6 channels, truncating title ...
    combining channels using "mean"
    combining channels using "mean"
    combining channels using "mean"
    combining channels using "mean"





![png](Project_2_files/Project_2_61_1.png)



Below, do the same thing, but for the input data specify a list with two entries: from `evoked_by_cond`, use the two conditions in the `'WPtn-Ctrl'` entry in `contrasts`. Using `evoked_by_cond`, MNE will generate a waveform plot with 95% confidence intervals shading.


```python
# List: list with two entries from evoked_by_condition
lis=[]

a = evoked_by_cond['Ctrl']
b = evoked_by_cond['WPtn']

lis.append(a)
lis.append(b)

lis
```




    [[<Evoked  |  'Ctrl' (average, N=30), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=34), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=13), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=16), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=29), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=24), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'Ctrl' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>],
     [<Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=36), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=37), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=19), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=33), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=38), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=17), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=28), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=35), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=39), [-0.19922, 1] sec, 64 ch, ~477 kB>,
      <Evoked  |  'WPtn' (average, N=31), [-0.19922, 1] sec, 64 ch, ~477 kB>]]




```python
#Plot using evoked_by_cond: 

mne.viz.plot_compare_evokeds(lis, 
                             picks=rois['Mmid'],
                             combine='mean',
                             #linestyles=['--','-'],
                             show_sensors=4);
plt.show()
```

    More than 6 channels, truncating title ...
    combining channels using "mean"
    combining channels using "mean"





![png](Project_2_files/Project_2_64_1.png)



Now do the same thing as in the last cell, but looping through all of the contrasts and generating a plot for each. Be sure each plot has a title indicating the contrast.


```python
# Plots for each contrast 
for g in gavg_diff:
     mne.viz.plot_compare_evokeds(gavg_diff[g],
                             picks=rois['Mmid'],
                             combine='mean',
                             linestyles= None,
                             show_sensors=4,
                             title= g
                                );
        
plt.show()
```

    combining channels using "mean"





![png](Project_2_files/Project_2_66_1.png)



    combining channels using "mean"





![png](Project_2_files/Project_2_66_3.png)



    combining channels using "mean"





![png](Project_2_files/Project_2_66_5.png)



    combining channels using "mean"





![png](Project_2_files/Project_2_66_7.png)



---
## Plot mean amplitudes over time windows of interest

While we can eyeball the differences using topomaps and waveform plots, averaging over the time windows of interest and plotting the means as box plots or bar graphs can be useful as well. This is easy to do using Seaborn, as you have done in the past with data in pandas DataFrames. Fortunately, MNE provide the `Evoked.to_data_frame()` method to convert MNE Evoked data to a pandas DataFrame. 

### Extract data to Pandas dataframe
In the cell below, write a loop to cycle through subjects and conditions and extract the data from each as a pandas DataFrame. Append these to `df_list` and then at the end, concatenate them into one DataFrame. Use the following arguments to `Evoked.to_data_frame`: `time_format='ms', picks=rois['Mmid'], long_format=True`. Be sure to add columns that encode which subject and condition the data corresponds to, as these will not be created by default. 


<font color='green'>
    <h2>HINT:</h2> 
    You're still not looping through conditions, but you're looking through subjects twice, which is wrong.


```python


# Initalize empty list
df_listed = []

# Loop through subjects in evoked dictionary
for subject in evoked:

    # Loop through condition list within subjects
    for i,x in enumerate(evoked[subject]):
        # Convert evoked to DataFrame
        df = mne.Evoked.to_data_frame(x, time_format='ms', picks=rois['Mmid'], long_format=True)

        # Add columns to specify subject and condition
        df['subject'] = subject
        df['condition'] = evoked[subject][i].comment

        # Add DataFrame to list
        df_listed.append(df)

# Concatenate the list
df_1 = pd.concat(df_listed, ignore_index=True)

```

    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...


Show the head and tail of the DataFrame:


```python
df_1.head()
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
      <th>time</th>
      <th>channel</th>
      <th>ch_type</th>
      <th>value</th>
      <th>subject</th>
      <th>condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-199</td>
      <td>C1</td>
      <td>eeg</td>
      <td>-0.606825</td>
      <td>MSA_01</td>
      <td>Ctrl</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-199</td>
      <td>Cz</td>
      <td>eeg</td>
      <td>-1.372333</td>
      <td>MSA_01</td>
      <td>Ctrl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-199</td>
      <td>C2</td>
      <td>eeg</td>
      <td>-0.112283</td>
      <td>MSA_01</td>
      <td>Ctrl</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-199</td>
      <td>CP1</td>
      <td>eeg</td>
      <td>-0.482514</td>
      <td>MSA_01</td>
      <td>Ctrl</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-199</td>
      <td>CPz</td>
      <td>eeg</td>
      <td>-1.566580</td>
      <td>MSA_01</td>
      <td>Ctrl</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_1.tail()
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
      <th>time</th>
      <th>channel</th>
      <th>ch_type</th>
      <th>value</th>
      <th>subject</th>
      <th>condition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>398515</th>
      <td>1000</td>
      <td>CPz</td>
      <td>eeg</td>
      <td>-2.307443</td>
      <td>MSA_19</td>
      <td>WPtn</td>
    </tr>
    <tr>
      <th>398516</th>
      <td>1000</td>
      <td>CP2</td>
      <td>eeg</td>
      <td>-1.356308</td>
      <td>MSA_19</td>
      <td>WPtn</td>
    </tr>
    <tr>
      <th>398517</th>
      <td>1000</td>
      <td>P1</td>
      <td>eeg</td>
      <td>-3.410524</td>
      <td>MSA_19</td>
      <td>WPtn</td>
    </tr>
    <tr>
      <th>398518</th>
      <td>1000</td>
      <td>Pz</td>
      <td>eeg</td>
      <td>-3.148974</td>
      <td>MSA_19</td>
      <td>WPtn</td>
    </tr>
    <tr>
      <th>398519</th>
      <td>1000</td>
      <td>P2</td>
      <td>eeg</td>
      <td>-1.279431</td>
      <td>MSA_19</td>
      <td>WPtn</td>
    </tr>
  </tbody>
</table>
</div>



### Compute mean amplitude over a time window

Create a new DataFrame called `means` that contains the mean ERP amplitude value for each subject, channel, and condition, averaged over each time window of interest (N400 and P600). You will need to add a column indicating the time window (component) label, and you'll want to drop the original 'time' column as it will no longer be informative. 


```python
#time windows 400-600 and 700-900

#created two new dfs
df_N400 = df_1[(df_1['time'] >= 400) & (df_1['time'] <= 600)].copy()
df_P600 = df_1[(df_1['time'] >=700) & (df_1['time'] <= 900)].copy()

#new column in each
df_N400['t_window']= 'N400'
df_P600['t_window']= 'P600'

#concat and drop time coloumn
test_1 = pd.concat([df_N400,df_P600], axis=0, ignore_index=True).copy()
test_1 = test_1.drop(['time'], axis=1).copy()
#grouping means 
means = test_1.groupby(['t_window', 'subject', 'condition', 'channel'])[['value']].mean()

means
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
      <th></th>
      <th></th>
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>t_window</th>
      <th>subject</th>
      <th>condition</th>
      <th>channel</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">N400</th>
      <th rowspan="5" valign="top">MSA_01</th>
      <th rowspan="5" valign="top">Ctrl</th>
      <th>C1</th>
      <td>2.364423</td>
    </tr>
    <tr>
      <th>C2</th>
      <td>2.662286</td>
    </tr>
    <tr>
      <th>CP1</th>
      <td>0.794368</td>
    </tr>
    <tr>
      <th>CP2</th>
      <td>1.406106</td>
    </tr>
    <tr>
      <th>CPz</th>
      <td>1.967333</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">P600</th>
      <th rowspan="5" valign="top">MSA_19</th>
      <th rowspan="5" valign="top">WPtn</th>
      <th>CPz</th>
      <td>0.233629</td>
    </tr>
    <tr>
      <th>Cz</th>
      <td>0.719439</td>
    </tr>
    <tr>
      <th>P1</th>
      <td>0.018339</td>
    </tr>
    <tr>
      <th>P2</th>
      <td>-0.142229</td>
    </tr>
    <tr>
      <th>Pz</th>
      <td>-0.165535</td>
    </tr>
  </tbody>
</table>
<p>1296 rows × 1 columns</p>
</div>



Show the head and tail of `means`:


```python
means.head()
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
      <th></th>
      <th></th>
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>t_window</th>
      <th>subject</th>
      <th>condition</th>
      <th>channel</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">N400</th>
      <th rowspan="5" valign="top">MSA_01</th>
      <th rowspan="5" valign="top">Ctrl</th>
      <th>C1</th>
      <td>2.364423</td>
    </tr>
    <tr>
      <th>C2</th>
      <td>2.662286</td>
    </tr>
    <tr>
      <th>CP1</th>
      <td>0.794368</td>
    </tr>
    <tr>
      <th>CP2</th>
      <td>1.406106</td>
    </tr>
    <tr>
      <th>CPz</th>
      <td>1.967333</td>
    </tr>
  </tbody>
</table>
</div>




```python
means.tail()
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
      <th></th>
      <th></th>
      <th></th>
      <th>value</th>
    </tr>
    <tr>
      <th>t_window</th>
      <th>subject</th>
      <th>condition</th>
      <th>channel</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">P600</th>
      <th rowspan="5" valign="top">MSA_19</th>
      <th rowspan="5" valign="top">WPtn</th>
      <th>CPz</th>
      <td>0.233629</td>
    </tr>
    <tr>
      <th>Cz</th>
      <td>0.719439</td>
    </tr>
    <tr>
      <th>P1</th>
      <td>0.018339</td>
    </tr>
    <tr>
      <th>P2</th>
      <td>-0.142229</td>
    </tr>
    <tr>
      <th>Pz</th>
      <td>-0.165535</td>
    </tr>
  </tbody>
</table>
</div>



## Seaborn plots

Generate a boxplot of values from `means`, with amplitude on the *y* axis and condition on the *x* axis. Use separate columns for the two component time windows.


```python
#reset index to make plotting easier 
means = means.reset_index()
```


```python
# Create colour blind friendly pallette
pal = ['#F5793A', '#A95AA1', '#85C0F9', '#0F2080']
```


```python
sns.set_palette(pal)
sns.set_style('whitegrid')

#create boxplot
fig1 = sns.boxplot(x='condition', y='value', hue='t_window', data=means)
fig1.set_title('Mean amplitude per condition and time window', fontsize=15)
fig1.set_xlabel('Condition', fontsize=13)
fig1.set_ylabel('Amplitude (µV)', fontsize=13)
plt.legend(title='Time window', title_fontsize=13, fontsize=13)
plt.show()
```




![png](Project_2_files/Project_2_81_0.png)



Do the same as above, but a violin plot:


```python
sns.set_palette(pal)
sns.set_style('whitegrid')

#create violin plot
fig2 = sns.violinplot(x='condition', y='value', hue='t_window', data=means)
fig2.set_title('Mean amplitude per condition and time window', fontsize=15)
fig2.set_xlabel('Condition', fontsize=13)
fig2.set_ylabel('Amplitude (µV)', fontsize=13)
plt.legend(title='Time window', title_fontsize=13, fontsize=13)
plt.show()
```




![png](Project_2_files/Project_2_83_0.png)



Do the same as above, but a bar plot:


```python
sns.set_palette(pal)
sns.set_style('whitegrid')

#create bar plot
fig3 = sns.barplot(x='condition', y='value', hue='t_window', data=means, ci='sd', errwidth=0.8)
fig3.set_title('Mean Amplitude per Condition and Time window', fontsize=15)
fig3.set_xlabel('Condition', fontsize=13)
fig3.set_ylabel('Amplitude (µV)', fontsize=13)
plt.legend(title='Time window', title_fontsize=13, fontsize=13)
plt.show()
```




![png](Project_2_files/Project_2_85_0.png)



## Plot differences

Follow the same steps as above to extract teh MNE data to a pandas DataFrame, then create averages within each component time window, and plot. This time, however, do this for each *contrast* rather than each condition. 


```python
# Initalize empty list
contr_listed = []

# Loop through subjects in evoked dictionary, creating df for each contrast
for subject in evoked:
    ctrl = evoked[subject][0]
    rnots = evoked[subject][1]
    rpluss = evoked[subject][2]
    wptn = evoked[subject][3]

    weights=[1, -1]

    temp1 = []
    temp1.append(wptn)
    temp1.append(ctrl)
    w_c = mne.combine_evoked(temp1, weights)

    temp2 = []
    temp2.append(rpluss)
    temp2.append(ctrl)
    rp_c = mne.combine_evoked(temp2, weights)

    temp3 = []
    temp3.append(rnots)
    temp3.append(ctrl)
    rn_c = mne.combine_evoked(temp3, weights)

    temp4 = []
    temp4.append(rpluss)
    temp4.append(rnots)
    rp_rn = mne.combine_evoked(temp4, weights)

    df1 = mne.Evoked.to_data_frame(w_c, time_format='ms', picks=rois['Mmid'], long_format=True)
    df1['subject']=subject
    df1['contrast']='Wptn-Ctrl'

    df2 = mne.Evoked.to_data_frame(rp_c, time_format='ms', picks=rois['Mmid'], long_format=True)
    df2['subject']=subject
    df2['contrast']='RplusS-Ctrl'

    df3 = mne.Evoked.to_data_frame(rn_c, time_format='ms', picks=rois['Mmid'], long_format=True)
    df3['subject']=subject
    df3['contrast']='RnotS-Ctrl'

    df4 = mne.Evoked.to_data_frame(rp_rn, time_format='ms', picks=rois['Mmid'], long_format=True)
    df4['subject']=subject
    df4['contrast']='RplusS-RnotS'

    #create one df per subject 
    df_list = [df1, df2, df3, df4]
    df_per_sub = pd.concat(df_list, ignore_index=True)

    contr_listed.append(df_per_sub)

#concat dfs from all subjects
df_contr = pd.concat(contr_listed, ignore_index=True)

```

    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...
    Converting "channel" to "category"...
    Converting "ch_type" to "category"...





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
      <th>time</th>
      <th>channel</th>
      <th>ch_type</th>
      <th>value</th>
      <th>subject</th>
      <th>contrast</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-199</td>
      <td>C1</td>
      <td>eeg</td>
      <td>0.295124</td>
      <td>MSA_01</td>
      <td>Wptn-Ctrl</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-199</td>
      <td>Cz</td>
      <td>eeg</td>
      <td>1.462550</td>
      <td>MSA_01</td>
      <td>Wptn-Ctrl</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-199</td>
      <td>C2</td>
      <td>eeg</td>
      <td>-0.041921</td>
      <td>MSA_01</td>
      <td>Wptn-Ctrl</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-199</td>
      <td>CP1</td>
      <td>eeg</td>
      <td>0.180475</td>
      <td>MSA_01</td>
      <td>Wptn-Ctrl</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-199</td>
      <td>CPz</td>
      <td>eeg</td>
      <td>1.174346</td>
      <td>MSA_01</td>
      <td>Wptn-Ctrl</td>
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
      <th>398515</th>
      <td>1000</td>
      <td>CPz</td>
      <td>eeg</td>
      <td>-0.294647</td>
      <td>MSA_19</td>
      <td>RplusS-RnotS</td>
    </tr>
    <tr>
      <th>398516</th>
      <td>1000</td>
      <td>CP2</td>
      <td>eeg</td>
      <td>-1.669542</td>
      <td>MSA_19</td>
      <td>RplusS-RnotS</td>
    </tr>
    <tr>
      <th>398517</th>
      <td>1000</td>
      <td>P1</td>
      <td>eeg</td>
      <td>0.747168</td>
      <td>MSA_19</td>
      <td>RplusS-RnotS</td>
    </tr>
    <tr>
      <th>398518</th>
      <td>1000</td>
      <td>Pz</td>
      <td>eeg</td>
      <td>0.528971</td>
      <td>MSA_19</td>
      <td>RplusS-RnotS</td>
    </tr>
    <tr>
      <th>398519</th>
      <td>1000</td>
      <td>P2</td>
      <td>eeg</td>
      <td>0.657723</td>
      <td>MSA_19</td>
      <td>RplusS-RnotS</td>
    </tr>
  </tbody>
</table>
<p>398520 rows × 6 columns</p>
</div>



### Compute mean amplitude over a time window

For each subject, channel, condition


```python
#select time windows
df_N400_dif = df_contr[(df_contr['time'] >= 400) & (df_contr['time'] <= 600)].copy()
df_P600_dif = df_contr[(df_contr['time'] >=700) & (df_contr['time'] <= 900)].copy()

#new coloumn in each
df_N400_dif['t_window']= 'N400'
df_P600_dif['t_window']= 'P600'

#concat and drop time coloumn
contr_df = pd.concat([df_N400_dif,df_P600_dif], axis=0, ignore_index=True).copy()
contr_df = contr_df.drop(['time'], axis=1).copy()

#grouping means 
means_contr = contr_df.groupby(['t_window','contrast','channel', 'subject'])[['value']].mean()

#reset index for easier plotting
means_contr = means_contr.reset_index()
```

### Now plot
Generate Seaborn box, violin, and bar plots of the contrasts.


```python
sns.set_palette(pal)
sns.set_style('whitegrid')

#create boxplot
fig4 = sns.boxplot(x='contrast', y='value', hue='t_window', data=means_contr)
fig4.set_title('Mean Amplitude per Contrast and Time Window', fontsize=15)
fig4.set_xlabel('Contrast', fontsize=13)
fig4.set_ylabel('Amplitude (µV)', fontsize=13)
plt.legend(title='Time window', title_fontsize=13, fontsize=13, loc='upper left')
plt.show()
```




![png](Project_2_files/Project_2_91_0.png)




```python
sns.set_palette(pal)
sns.set_style('whitegrid')

#create violinplot
fig5 = sns.violinplot(x='contrast', y='value', hue='t_window', data=means_contr)
fig5.set_title('Mean Amplitude per Contrast and Time Window', fontsize=15)
fig5.set_xlabel('Contrast', fontsize=13)
fig5.set_ylabel('Amplitude (µV)', fontsize=13)
plt.legend(title='Time window', title_fontsize=13, fontsize=13)
plt.show()
```




![png](Project_2_files/Project_2_92_0.png)




```python
sns.set_palette(pal)
sns.set_style('whitegrid')

#create bar plot
fig6 = sns.barplot(x='contrast', y='value', hue='t_window', data=means_contr, ci='sd', errwidth=0.8)
fig6.set_title('Mean Amplitude per Contrast and Time Window', fontsize=15)
fig6.set_xlabel('Contrast', fontsize=13)
fig6.set_ylabel('Amplitude (µV)', fontsize=13)
plt.legend(title='Time window', title_fontsize=13, fontsize=13, loc='lower left')
plt.show()
```




![png](Project_2_files/Project_2_93_0.png)



---
# Stats

## t-tests over specific time windows

Use `scipy.stats.ttest_rel` to perform a pairwise *t*-test between conditions for each of the experimental contrasts, averaged across the electrodes in the `Mmid` ROI, in each of the two component time windows (N400 and P600). Report the *t* and *p* values for each contrast.


```python
# Initialize time window names

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
    


## Interpretation

Using the evidence produced above, state whether each hypothesis was supported or not.

### WPtn - Ctrl: N400

Based on our analyses the first hypothesis (WPtn - Ctrl: N400) was supported by the data. As seen in this topographic map 

![topo (Figure 1)](WPtn_Ctrl_N400.png)




 negative ERPs occur, especially in the Mmid ROI at 0.400s. We can also see a more negative potential in the waveform graph with a peak at around 0.45s. 
 
![wave (Figure 1)](WPtn_Ctrl_wave.png) 


In fact, the voltage in the waveform graph remains negative until just after the 0.6s mark. This is consistent with our selection of 400- 600ms as our time window for N400. 


By examining the bar plots(figure 1), we can also see that this contrast had the majority of data at the time below 0 during the N400 window, which suggests a difference between WPtnS and Ctrl conditions. However, this contrast did have the most variability suggesting that further experimentation should be conducted to increase the power of our test. 

Lastly, our t-test indicates a significant effect for the N400 ERP for contrast Wptn-Ctrl (t=-22.656, p=0.00). Overall, our analyses tend to support our hypotheses, but, given our discussion on the variability of our data, we encourage more data collection. 


### RplusS - Ctrl: N400 & P600

The second hypothesis (RplusS - Ctrl: N400 & P600) was also supported by our analyses. And the conditions elicited both a N400 and P600. This can be seen in the activity in the topographic maps for the time windows as well as the waveform graphs.


![topo (Figure 1)](RplusS_Ctrl_topo_N400.png)

![wave (Figure 1)](RnotS_Ctrl_wave.png)



The box plots (figure 2) show that there is a negative amplitude occurring in the N400 time window and that there is some positive spiking happening in the P600 window, even though the positive spiking are not that prevalent compared to the N400. These results also came back significant in the t-test (N400: t=-15.086, p=0.00; P600: t=3.468, p=0.001). 





### RnotS - Ctrl: N400 & P600



The data suggest that the third hypothesis (RnotS - Ctrl: N400 & P600) was supported because there was no evidence of a P600 occurring in the averaged data. However there was evidence of a N400 occurring instead. This can be seen in the waveform, with a negative amplitude peaking just after 0.4s and a positive amplitude following 0.6s. This is supported with the topographic maps, which show amplitudes consistent with the waveform graph. 



![boxplots (Figure 1)](RnotS_Ctrl_topo_N400.png)

![boxplots (Figure 1)](RnotS_Ctrl_topo_P600.png)

![boxplots (Figure 1)](RnotS_Ctrl_wave.png)
This is consistent with the results of our t-test (N400: t=-19.271, p=0.000; P600: t(199)=2.600, p=0.009), which show significant effects for the N400 and P600. 

### RplusS-RnotS: N400

The fourth hypothesis (RplusS-RnotS: N400) was not supported. By looking at the waveform there was no clear indication of a N400 happening (N400:t(199)=1.891, p=0.059;P600: t(199)=1.109, p=0.267)The box plots(figure 2) even show that there is a slight positive mean during this time period. 

![boxplots (Figure 1)](RplusS_RnotS_topo_N400.png)

![boxplots (Figure 1)](RplusS_RnotS_wave.png)



## Other Figures referenced  


![barplot (Figure 1)](mean_box_plot.png)
### *(figure 1)*




![boxplots (Figure 1)](mean_bar_plot.png)
### *(figure 2)*

# Contributions

Below, list the contributions each team member made to the project:


1. Maya Dickson
    1. Troubleshooting for reading in data
    2. Difference waves
    3. Topographic maps
2. Sarah Harrison
    1. Seaborn plots
    2. Plot differences
3. Amara Huntington
    1. A priori time windows of interest
4. Arlene Jiang
    1. Stats
    2. Interpretation
5. Alexander Kotzeff
    1. Reading in data
    2. First part of Experimental Conditions and Contrasts to check the data.
    3. Troubleshooting for gavg and evoked_by_cond dictionaries
6. Reann Post
    1. Extract data to pandas df
    2. Compute mean amplitude over a time window
7. Chloe Robichaud
    1. Experimental Conditions and Contrast
    2. Grand Averages
8. Meg South
    1. Waveform plots
9. Isaac Zacher
    1. Stats
    2. Interpretation
    3. helped with gavg 
    4. helped to convert to df 
    4. helped to make contrast df 
