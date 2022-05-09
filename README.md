---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.8
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="bWit1CuWK0Ik" -->
#Importing basic libraries and loading data sets
<!-- #endregion -->

```python id="k9UD-upEsd85"
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline

# To manage ticker in plot
from matplotlib.ticker import MaxNLocator
```

```python colab={"base_uri": "https://localhost:8080/", "height": 206} id="ZpGmYrw-tEY-" outputId="fa310af2-7b41-4a19-e71d-e2f4598dcecb"
train_data=pd.read_csv('/content/train.csv')
train_data.head(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="HpIFr189wa_z" outputId="ca7a3257-6d4c-41c8-863e-359078cf3378"
print('number of passenger in training data:'+str(len(train_data.index)))
```

```python colab={"base_uri": "https://localhost:8080/"} id="KUKVClkUzSG2" outputId="f2df18d8-47a8-4bb3-f275-999993c33326"
test_data=pd.read_csv('/content/test.csv')
print('number of passenger in testing data:'+str(len(test_data.index)))
```

<!-- #region id="aGK2teFQxa4t" -->
#Analyzing Data 
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 35} id="XEQfR5kzxfcd" outputId="81327c30-d2fc-416d-a1e5-e8a2740bba08"
# Let pandas show all columns
pd.options.display.width = 0

# For better viusalisation increase horizontal space for subplots
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.4)

# Seaborn theme setings
sns.set_theme(style = "whitegrid", palette="deep")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 561} id="SdPErgMRBSbE" outputId="7342e91d-e096-4086-d20a-c8cf4c4c1450"
#merging training and testing data for missing values imputation
len_train=len(train_data)
data_all=pd.concat(objs=[train_data,test_data],axis=0).reset_index(drop=True)
data_all.info()
print('number of passenger in all data:'+str(len(data_all.index)))
data_all.head()
```

```python colab={"base_uri": "https://localhost:8080/"} id="LX3MwZUfRaE8" outputId="5273b200-6cab-4dcb-cd55-5332253184e8"
#check missing data
null_data=data_all.isnull().sum()
null_data[null_data>0]
```

```python colab={"base_uri": "https://localhost:8080/"} id="2wb3CPqnO8Uu" outputId="39e915cc-6c2b-49b4-99ae-8fae1bc006b3"
#Converting some columns to categories and category labels to discrete numbers
data_all["Cabin_Group"] = data_all["Cabin"].str[:1]
data_all["Cabin_Group"] = data_all["Cabin_Group"].astype('category')
cabin_group_categories = dict(enumerate(data_all["Cabin_Group"].cat.categories))

data_all["Cabin_Group"] = data_all["Cabin_Group"].cat.codes
data_all["Cabin_Group"] = data_all["Cabin_Group"].astype(int)

data_all["Sex"] = data_all["Sex"].astype('category')
sex_categories = dict(enumerate(data_all["Sex"].cat.categories))
data_all["Sex"] = data_all["Sex"].cat.codes
data_all["Sex"] = data_all["Sex"].astype(int)

data_all["Embarked"] = data_all["Embarked"].astype('category')
embarked_categories = dict(enumerate(data_all["Embarked"].cat.categories))
data_all["Embarked"] = data_all["Embarked"].cat.codes
data_all["Embarked"] = data_all["Embarked"].astype(int)

data_all = data_all.convert_dtypes()
data_all.info()
```

```python colab={"base_uri": "https://localhost:8080/"} id="Yh9jIKZFUK8Z" outputId="f0b0a841-d72f-49ce-fa86-d25aa547e8be"
print('\nEmbarked:')
print(embarked_categories)
print('\nCabin Group:')
print(cabin_group_categories)
print('\nSex:')
print(sex_categories)
```

<!-- #region id="yUttTxSTSydZ" -->
PLOTS and imputing missing data
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="twyp4M09M0c5" outputId="49a09ba8-c5c7-426e-945e-620c5a55e4fe"
#fare
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="Fare")
sns.scatterplot(ax=axes[0,1], data=data_all, x="Fare", y="Survived")
sns.scatterplot(ax=axes[0,2], data=data_all, x="Fare", y="Pclass")
sns.scatterplot(ax=axes[1,0], data=data_all, x="Fare", y="Age")
sns.scatterplot(ax=axes[1,1], data=data_all, x="Fare", y="Sex")
sns.scatterplot(ax=axes[1,2], data=data_all, x="Fare", y="SibSp")
sns.scatterplot(ax=axes[2,0], data=data_all, x="Fare", y="Parch")
sns.scatterplot(ax=axes[2,1], data=data_all, x="Fare", y="Cabin_Group")
sns.scatterplot(ax=axes[2,2], data=data_all, x="Fare", y="Embarked")

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 81} id="o_OvZZN6F9Jv" outputId="5fe11111-ceec-44c9-a17d-db8cda273bb5"
#checking the one missing fare data
data_all[data_all["Fare"].isnull()]
```

```python id="COQmaG4-JXy9"
#filling the one missing fare data
m = data_all[data_all["Pclass"]==3]["Fare"].median()
data_all["Fare"] = data_all["Fare"].fillna(m)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="oaDK-SB9Jly_" outputId="f1fd9ee4-d680-4f7c-e25a-fe5852945f6f"
#embarked 
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

# Make ticks integer for discrete values
axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,0].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="Embarked")
sns.boxplot(ax=axes[0,1], data=data_all, x="Embarked",y="Survived")
sns.boxplot(ax=axes[0,2], data=data_all, x="Embarked",y="Pclass")
sns.boxplot(ax=axes[1,0], data=data_all, x="Embarked",y="Sex")
sns.boxplot(ax=axes[1,1], data=data_all, x="Embarked",y="Age")
sns.boxplot(ax=axes[1,2], data=data_all, x="Embarked",y="SibSp")
sns.boxplot(ax=axes[2,0], data=data_all, x="Embarked",y="Parch")
sns.boxplot(ax=axes[2,1], data=data_all, x="Embarked",y="Fare")
sns.boxplot(ax=axes[2,2], data=data_all, x="Embarked",y="Cabin_Group")

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 112} id="aMteOzChXytZ" outputId="5822d07b-7e2e-4ba6-da46-0eb0d8dc8a81"
#checking missing embarked values
data_all[data_all["Embarked"]==-1]
```

<!-- #region id="SjmTTStETarr" -->

Fare column shows us Embarked is "C" and and Pclass column confirms it Also we see that Cabin Names start with B belongs to port 0 (= C) We can fill missing Embarked value with "0"

Most frequent embarkation point is 2(=S).

Pclass at embarkation point 1(=Q) is 3 with some outliers. Embarkation point 2(=S) majors on Pclass 2 and 3.

The people at embarkation point 1(=Q) are somewhat younger than the others. Also most of them possibly do not have a family.

The embarkation point 0(=C) mostly consists of families with children. In every Pclass, there are children.

The people at embarkation point 0(=C) pays more for tickets.

Cabin_Group 0(=A) and 1(=B) mostly uses embarkation point 0(=C)
<!-- #endregion -->

```python id="pu0qBkKFTAco"
#imputing missing embarked values
data_all["Embarked"] = data_all["Embarked"].replace(-1,0)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="VNYxrmpxYCtT" outputId="ca619349-25aa-47d1-dfa6-ae03ce05b233"
#cabin group
fig,axes = plt.subplots(nrows=3, ncols=3,figsize=(15,15))

axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,0].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="Cabin_Group")
sns.boxplot(ax=axes[0,1], data=data_all, x="Cabin_Group", y="Survived")
sns.boxplot(ax=axes[0,2], data=data_all, x="Cabin_Group", y="Pclass")
sns.boxplot(ax=axes[1,0], data=data_all, x="Cabin_Group", y="Sex")
sns.boxplot(ax=axes[1,1], data=data_all, x="Cabin_Group", y="Age")
sns.boxplot(ax=axes[1,2], data=data_all, x="Cabin_Group", y="SibSp")
sns.boxplot(ax=axes[2,0], data=data_all, x="Cabin_Group", y="Parch")
sns.boxplot(ax=axes[2,1], data=data_all, x="Cabin_Group", y="Fare")
sns.boxplot(ax=axes[2,2], data=data_all, x="Cabin_Group", y="Embarked")

plt.show()
```

<!-- #region id="wPq5EKS8YjzQ" -->
A high amount of Cabin_Group data is missing.

Except missing data: Cabin_Group 1(=B), 3(=D), 4(=E) was mostly survived and Cabin_Group 7(=T) did not survive.

Missing Cabin_Group data mostly belongs to Pclass 2 and 3. While Cabin_Group 5(=F) belongs to Pclass 2 and 3, it is 3 for Cabin_Group 6(=G)

People at Cabin_Group 0(=A) and 7(=T) is male, it is female for Cabin_Group 6(=G)

Age range is mostly 20-35 for missing Cabin_Group data.

Cabin_Group 0(=A) consists of older people. Young ones is at Cabin_Group 5(=F) and Cabin_Group 6(=G)

Except Cabin_Group 0(=A) and 7(=T), SibSp is not descriptive for Cabin_Group. It is similar for Parch.

People at Cabin_Group 2(=C) pays more for tickets.

Missing Cabin_Group data majors on embarkation point 2(=S)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="4W_1TSTeYWGw" outputId="ad833b84-45ef-4aef-a4c0-462d2905b1c3"
#imputing missing values in cabin group
missing_survived = len(data_all[(data_all['Cabin_Group']==-1) &(data_all["Survived"]==1)])
missing_not_survived = len(data_all[(data_all['Cabin_Group']==-1) &(data_all["Survived"]==0)])

cabin_null_count = len(data_all[(data_all['Cabin_Group']==-1)])
                               
print("Survived Percentage in Missing Cabin Values : ", '{:.0%}'.format(missing_survived / cabin_null_count))
print("Not-Survived Percentage in Missing Cabin Values : ", '{:.0%}'.format(missing_not_survived/ cabin_null_count))
```

<!-- #region id="V6NR2VU-J1ks" -->
Since cabin_group gives good survival prediction so we will keep cabin_group and drop cabin
<!-- #endregion -->

```python id="7qy2eE-FJjBJ"
data_all = data_all.drop(['Cabin'], axis=1)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="CLYaBhglM86R" outputId="1049fb98-4837-4426-f295-6e506bab90a6"
#age
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

sns.histplot(ax=axes[0,0], data=data_all, x="Age",bins=10)
sns.boxplot(ax=axes[0,1], data=data_all, x="Pclass", y="Age")
sns.boxplot(ax=axes[0,2], data=data_all, x="Sex", y="Age")
sns.boxplot(ax=axes[1,0], data=data_all, x="SibSp", y="Age")
sns.boxplot(ax=axes[1,1], data=data_all, x="Parch", y="Age")
sns.scatterplot(ax=axes[1,2], data=data_all, x="Fare", y="Age")
sns.boxplot(ax=axes[2,0], data=data_all, x="Cabin_Group", y="Age")
sns.boxplot(ax=axes[2,1], data=data_all, x="Embarked", y="Age")
fig.delaxes(axes[2,2])

plt.show()
```

<!-- #region id="416gAtsKR-hP" -->
In the light of the above inspection we will use Pclass, SibSp, Parch, Cabin_Group, Survived and Embarked features to impute Age feature using decision tree.
<!-- #endregion -->

```python id="3LPNVyPZR0eK"
#imputing missing age
from sklearn.tree import DecisionTreeRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data_impute_dtree = data_all.copy()
data_impute_dtree = data_impute_dtree.drop(["Name", "PassengerId","Ticket", "Sex","Fare"], axis=1)

dtr = DecisionTreeRegressor()
imp = IterativeImputer(estimator=dtr, missing_values=np.nan, max_iter=500, verbose=0, imputation_order='roman', random_state=42, min_value=0)
x_imputed = imp.fit_transform(data_impute_dtree)

data_impute_dtree["MF_Age"] = x_imputed[:,2]
data_impute_dtree["MF_Age"] = data_impute_dtree["MF_Age"].astype(int)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="9fjWuONBKPMS" outputId="66e40361-380d-4a05-d135-5d26f3453828"
#Pclass
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="Pclass")
sns.boxplot(ax=axes[0,1], data=data_all, x="Pclass", y="Survived")
sns.boxplot(ax=axes[0,2], data=data_all, x="Pclass", y="Sex")
sns.boxplot(ax=axes[1,0], data=data_all, x="Pclass", y="Age")
sns.boxplot(ax=axes[1,1], data=data_all, x="Pclass", y="SibSp")
sns.boxplot(ax=axes[1,2], data=data_all, x="Pclass", y="Parch")
sns.boxplot(ax=axes[2,0], data=data_all, x="Pclass", y="Fare")
sns.boxplot(ax=axes[2,1], data=data_all, x="Pclass", y="Cabin_Group")
sns.boxplot(ax=axes[2,2], data=data_all, x="Pclass", y="Embarked")

plt.show()
```

<!-- #region id="oFbDcdycLONL" -->
Pclass 3 mostly not survived.

Pclass 1 consists of older people. Younger people is at Pclass 3.

Families are mostly at Pclass 2.

Pclass 1 pays more for tickets.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="s_Zf4CYWLKjc" outputId="26f94042-f369-441f-af6b-26aa30fca54a"
#sex vs others
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="Sex")
sns.boxplot(ax=axes[0,1], data=data_all, x="Sex", y="Survived")
sns.boxplot(ax=axes[0,2], data=data_all, x="Sex", y="Pclass")
sns.boxplot(ax=axes[1,0], data=data_all, x="Sex", y="Age")
sns.boxplot(ax=axes[1,1], data=data_all, x="Sex", y="SibSp")
sns.boxplot(ax=axes[1,2], data=data_all, x="Sex", y="Parch")
sns.boxplot(ax=axes[2,0], data=data_all, x="Sex", y="Fare")
sns.boxplot(ax=axes[2,1], data=data_all, x="Sex", y="Cabin_Group")
sns.boxplot(ax=axes[2,2], data=data_all, x="Sex", y="Embarked")

plt.show()

```

<!-- #region id="adxeoCQ8Lz2C" -->
The number of men is (about) twice the number of women.

The female mostly survived.

Males major at Pclass 2 and 3.

Females are slightly younger.

Mostly females have Sibling/Spouse/Children.

Females pay more for tickets.

Females mostly embarked at 1(=2) and 2(=S)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="5SVukKiJLxI4" outputId="07a71292-7116-4017-84df-f37294255fd0"
#SibSp(siblings or spouse on board)
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="SibSp")
sns.boxplot(ax=axes[0,1], data=data_all, x="SibSp", y="Survived")
sns.boxplot(ax=axes[0,2], data=data_all, x="SibSp", y="Pclass")
sns.boxplot(ax=axes[1,0], data=data_all, x="SibSp", y="Age")
sns.boxplot(ax=axes[1,1], data=data_all, x="SibSp", y="Sex")
sns.boxplot(ax=axes[1,2], data=data_all, x="SibSp", y="Parch")
sns.boxplot(ax=axes[2,0], data=data_all, x="SibSp", y="Fare")
sns.boxplot(ax=axes[2,1], data=data_all, x="SibSp", y="Cabin_Group")
sns.boxplot(ax=axes[2,2], data=data_all, x="SibSp", y="Embarked")

plt.show()
```

<!-- #region id="VVwlvsADL_Hx" -->
Most people have no sibling/spouse. If they have it is probably one.

More than 2 SibSp dramatically decreases survival possibility.

There are more survivors at PClass 1 (Middle aged, embarked at 1(=Q))

More than 2 sibling/spouse means you are young (The more siblings the less age.)
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 883} id="PzhK54tGL71w" outputId="39f0ab45-335e-463c-8b66-9f2f6cbc0d27"
#Parch(parents or children on board)
fig,axes = plt.subplots(nrows=3, ncols=3, figsize=(15,15))

axes[0,0].xaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[1,1].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[0,2].yaxis.set_major_locator(MaxNLocator(integer=True))
axes[2,2].yaxis.set_major_locator(MaxNLocator(integer=True))

sns.histplot(ax=axes[0,0], data=data_all, x="Parch")
sns.boxplot(ax=axes[0,1], data=data_all, x="Parch", y="Survived")
sns.boxplot(ax=axes[0,2], data=data_all, x="Parch", y="Pclass")
sns.boxplot(ax=axes[1,0], data=data_all, x="Parch", y="Age")
sns.boxplot(ax=axes[1,1], data=data_all, x="Parch", y="Sex")
sns.boxplot(ax=axes[1,2], data=data_all, x="Parch", y="SibSp")
sns.boxplot(ax=axes[2,0], data=data_all, x="Parch", y="Fare")
sns.boxplot(ax=axes[2,1], data=data_all, x="Parch", y="Cabin_Group")
sns.boxplot(ax=axes[2,2], data=data_all, x="Parch", y="Embarked")

plt.show()
```

<!-- #region id="J-ccsJjTMSBJ" -->
Parch has some relation with Age. It has certain characteristics. More Parent/Children means paying more to tickets.
<!-- #endregion -->

```python id="up79tfP5U1Ox"
data_final = data_impute_dtree.copy()
data_final["Age"] = data_final["MF_Age"]
data_final = data_final.drop("MF_Age", axis=1)
data_final["Name"] = data_all["Name"]
data_final["Sex"] = data_all["Sex"]
data_final["Fare"] = data_all["Fare"]
```

<!-- #region id="Y70uPkc4Ul_o" -->
#Feature Engineering
<!-- #endregion -->

<!-- #region id="uFzDgB3HVdDO" -->
Pclass:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 340} id="F98QA4v7MX7e" outputId="4ea0e9a8-5f92-4ca5-df99-9075c216a155"
#Pclass
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=60)
axes[1].tick_params('x', labelrotation=60)

sns.countplot(ax=axes[0], x="Pclass", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Pclass", y="Survived")

plt.show()
```

<!-- #region id="tknkRBcGU_oc" -->
Pclass is a good feature to predict survival.
<!-- #endregion -->

<!-- #region id="8SCaD_vtVum2" -->
Name:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="ctfAFVsQVyh7" outputId="6bf21494-c197-4ef6-89db-629d083c3d51"
data_final["Name"].head()
```

```python id="yyzWFsc9WBOH"
data_final["Title"] = [i.split(".")[0].split(",")[-1].strip() for i in data_final["Name"]]
```

```python colab={"base_uri": "https://localhost:8080/", "height": 397} id="EXnBWrQfWEgY" outputId="f648cf0b-1b28-4857-e623-6f10ad56d309"
#plotting titles in the name
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=60)
axes[1].tick_params('x', labelrotation=60)

sns.countplot(ax=axes[0], x="Title", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Title", y="Survived")

plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="40M5H7E3WRC_" outputId="29ccce93-6396-42c8-8a1b-f46439b01270"
data_final["Title"].unique()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 387} id="u4uQj4UAWca-" outputId="22b8dd68-d730-4c6b-f1f5-cd7ded7c62aa"
data_final["Title"] = data_final["Title"].replace(["Capt", "Col", "Don", "Dr", "Major", "Rev", "Sir", "Jonkheer"], "Rare_Male")
data_final["Title"] = data_final["Title"].replace(["Lady", "the Countess", "Dona", "Mme", "Ms", "Mlle"], "Rare_Female")

fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=45)
axes[1].tick_params('x', labelrotation=45)

sns.countplot(ax=axes[0], x="Title", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Title", y="Survived")

plt.show()
```

<!-- #region id="643rQ7lHWx6D" -->
The title in the name gives good prediction of survival so we can use it.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="SgY0G03FWrEG" outputId="3727a404-381a-4bd1-fbae-be512aa3c226"
#converting titles to categories and discrete values
data_final["Title"] = data_final["Title"].astype('category')
title_categories = dict(enumerate(data_final["Title"].cat.categories))
data_final["Title"] = data_final["Title"].cat.codes
data_final["Title"]  =data_final["Title"].astype(int)

data_final.drop(labels=["Name"], axis=1, inplace=True)
data_final["Embarked"] = data_final["Embarked"].astype('category')

title_categories
```

<!-- #region id="I1z8YIkPXL8g" -->
Sex:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 341} id="Zvrpij6CXKQC" outputId="bf998feb-bf35-4723-fcf4-7a44d74978f0"
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=45)
axes[1].tick_params('x', labelrotation=45)

sns.countplot(ax=axes[0], x="Sex", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Sex", y="Survived")

plt.show()
```

<!-- #region id="WQLbahXfXSZk" -->
Sex of the people is a good feature to predict survival.
<!-- #endregion -->

<!-- #region id="sYDKUr5NXZJP" -->
Age:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 729} id="vnkZF077XSFO" outputId="204d7d0e-92cb-42fe-efad-70e925eb6c7e"
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(30,15))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=90)
axes[1].tick_params('x', labelrotation=90)

sns.countplot(ax=axes[0], x="Age", data = data_final)
sns.scatterplot(ax=axes[1], data=data_final, x="Age", y="Survived",s=70)

plt.show()
```

<!-- #region id="FgtUXamoXdo4" -->
Age can be used to predict survival.
<!-- #endregion -->

<!-- #region id="SbjMt6g76Ts2" -->
Parch+SibSp=Fmly_count:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 346} id="62l3uJCbXQkF" outputId="71057c94-eeec-4305-aa74-f2219712f2a3"
data_final["Fmly_Count"] = data_final["SibSp"] + data_final["Parch"] + 1

fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=45)
axes[1].tick_params('x', labelrotation=45)

sns.countplot(ax=axes[0], x="Fmly_Count", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Fmly_Count", y="Survived")

plt.show()
```

<!-- #region id="i1S6H-p97ikt" -->
Fmly_count can be used to predict survival.
<!-- #endregion -->

<!-- #region id="tcRhYAZM7pWD" -->
Fare:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 351} id="9qy4s3497Pjv" outputId="2bf7a3fc-6b02-469d-d4e0-b7a77d94ad96"
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=45)
axes[1].tick_params('x', labelrotation=45)

sns.histplot(ax=axes[0], x="Fare", data=data_final)
sns.scatterplot(ax=axes[1], data=data_final,x ="Fare",y="Survived")

plt.show()
```

<!-- #region id="GVFff99u8VnM" -->
Fare can be used to predict
<!-- #endregion -->

<!-- #region id="INx67yQ38cYO" -->
Cabin_group:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 344} id="3UYWRQiN7tI2" outputId="9e3191a0-760f-4bab-ce77-7cacd67560c6"
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=45)
axes[1].tick_params('x', labelrotation=45)

sns.countplot(ax=axes[0], x="Cabin_Group", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Cabin_Group", y="Survived")

plt.show()
```

<!-- #region id="15eNV_Mt8mVv" -->
Cabin_group can be used for prediction
<!-- #endregion -->

<!-- #region id="4zbM_hVp8s3Z" -->
Embarked:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 341} id="1ZDHEM-v8jgA" outputId="abf8584a-f848-42a6-fef7-d00cf176f486"
fig,axes = plt.subplots(nrows=1, ncols=2, figsize=(10,5))
axes[1].yaxis.set_major_locator(MaxNLocator(integer=True))

axes[0].tick_params('x', labelrotation=45)
axes[1].tick_params('x', labelrotation=45)

sns.countplot(ax=axes[0], x="Embarked", data=data_final)
sns.boxplot(ax=axes[1], data=data_final, x="Embarked", y="Survived")

plt.show()
```

<!-- #region id="Ix_omE6_82UT" -->
Embarked doesn't gives us some meaningful insight to predict
<!-- #endregion -->

<!-- #region id="RsggMciYMiAa" -->
#Training and testing data


<!-- #endregion -->

```python id="YHWJm0OT8wiL"
#get dummies
data_final = pd.get_dummies(data_final, columns=["Pclass", "Title", "Sex", "Fmly_Count", "Cabin_Group", "Embarked"])
```

```python id="nhKlQ-Ux_oj6"
#dropping unnecessary
data_final.drop(labels=['SibSp','Parch'],axis=1,inplace=True)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 299} id="boWOXAoyAjYi" outputId="3d8db224-2c14-4231-a7ad-df4241316fd6"
data_final.head(5)
```

```python colab={"base_uri": "https://localhost:8080/"} id="_eZnVWWx9u9o" outputId="79e787c2-5155-4ea3-ea16-4f4bc5941373"
#splitting of testing and training data
train_data = data_final[:len_train]
test_data = data_final[len_train:]

train_data.info()
```

```python id="kE5ojHk8_a9t"
x_train=train_data.drop('Survived',axis=1)
from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
#y_train=pd.DataFrame( train_data["Survived"])
#y_train=y_train.to_records()
y_train = train_data["Survived"]
y_train = y_train.astype('int')


```

```python colab={"base_uri": "https://localhost:8080/"} id="g72w96GlBVvj" outputId="aca6bafe-31a7-4041-fba4-59476bcfe532"
print(y_train)
```

```python id="oMLPxcCUBTBS"
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(x_train), y_train, test_size=0.33, random_state=42)
```

<!-- #region id="Dicvq5Yr_JFn" -->
#Prediction using KNN
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="eWyzIB0wacKF" outputId="320ead9b-4947-4d13-c726-556733dd1c71"

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

knn_mod = KNeighborsClassifier()
knn_mod.fit(x_train,y_train)

y_prediction_train=knn_mod.predict(x_train)
y_prediction_test=knn_mod.predict(x_test)

a=confusion_matrix(y_train,y_prediction_train)
print(a)
# outcome values order in sklearn
tp, fn, fp, tn =a.reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_train,y_prediction_train,labels=[1,0])
print('Classification report : \n',matrix)

b=confusion_matrix(y_test,y_prediction_test)
print(b)
# outcome values order in sklearn
tp, fn, fp, tn =b.reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_test,y_prediction_test,labels=[1,0])
print('Classification report : \n',matrix)

```

<!-- #region id="yQg0KzL4xAiE" -->
#Prediction using logistic regression
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="yon0itNdGHBm" outputId="7646aa2a-3d32-46c9-d6ca-e0ef57bb02c2"
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg = LogisticRegression()
logreg.fit(x_train, y_train)

accuracy_train = round(logreg.score(x_train, y_train) * 100, 2) 
accuracy_test = round(logreg.score(x_test, y_test) * 100, 2)

print("Training Accuracy: % {}".format(accuracy_train))
print("Testing Accuracy: % {}".format(accuracy_test))


y_prediction_train=logreg.predict(x_train)
y_prediction_test=logreg.predict(x_test)

a=confusion_matrix(y_train,y_prediction_train)
print(a)
# outcome values order in sklearn
tp, fn, fp, tn =a.reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_train,y_prediction_train,labels=[1,0])
print('Classification report : \n',matrix)

b=confusion_matrix(y_test,y_prediction_test)
print(b)
# outcome values order in sklearn
tp, fn, fp, tn =b.reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)
# classification report for precision, recall f1-score and accuracy
matrix = classification_report(y_test,y_prediction_test,labels=[1,0])
print('Classification report : \n',matrix)


```

```python colab={"base_uri": "https://localhost:8080/"} id="mU3A5rMmJlTZ" outputId="505840b1-0947-4090-ed4f-c5a6aa05c7f0"
#prediction by sklearn
x_test1 = test_data.drop("Survived", axis=1)
x_test1 = StandardScaler().fit_transform(x_test1)

y_predict=logreg.predict(x_test1)
test_data["Survived"] = y_predict.tolist()

test_data = pd.read_csv('/content/test.csv')
test_data.set_index("PassengerId")
test_data["Survived"] = y_predict.tolist()
test_data[["PassengerId","Survived"]].to_csv("submission1.csv", index=False)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 655} id="guySqTmYMT3d" outputId="ffac2c5f-6a67-4827-993a-f9cb5ef508c9"
df=pd.read_csv("/content/submission1.csv")
df.head(100)
```
