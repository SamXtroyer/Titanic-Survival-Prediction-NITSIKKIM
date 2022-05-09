# Titanic-Survival-Prediction-NITSIKKIM

The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships. One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

Here we try to analyze which factors were more likely to contribute to the death of the passengers and predict who is more likely to survive depending on the features.

**About Data Sets** 

Here we are using two csv files.
1. train.csv: To analyse the data and fit the machine learning models.
2. test.csv: In this data we are predicting the survival according to the trained ML model.


```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline

# To manage ticker in plot
from matplotlib.ticker import MaxNLocator
```

Here we are importing basic libraries pandas, Numpy, seaborn etc..

```
train_data=pd.read_csv('/content/train.csv')
train_data.head(5)
```
![train_head](http://url/to/https://drive.google.com/file/d/1amt8SfryzVNlqRFHB3wEAT0p_NcVZX0-/view?usp=sharing)







