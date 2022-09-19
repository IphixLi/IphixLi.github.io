---
title: Datatalks ML zoomcamp
date: 2022-09-05 21:35:48
tags:
- Machine learning
- Datatalks
- study-notes
categories:
- Learning
toc: true
cover: /gallery/datatalks_ML.jpg
thumbnail: /gallery/datatalks_ML.jpg
---
## Introduction

This machine learning zoomcamp is an online-based machine learning course by Datatalks.Club. it is a learn-by-doing class teaching bread-and-butter skills and techniques in machine learning with projects.
<!-- more -->
The course primarily runs synchronously in cohorts for around 4 months which successful completion of 2 of 3 projects guaranting a certificate. The lecture slides, video and resources are freely available in [this course github](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp). I am taking fall cohort 2022 which officially started 5 September 2022.

in this post I will post course notes by weeks and links to my homeworks completions.

## Week 1 - Introduction to Machine Learning

<strong>introduction to ML</strong>

machine learning is the process in the process of extracting patterns from data. machine learning allow us to make models that can classify data given or predict for the future.

data can be separated into two types:
- features : information about the object such as attributes in dataset such as mileage, age, horsepower, location etc. these are the base data for making prediction
- target: this is the attribute we are developing model to predict for or classify against. for example it can be price of sales prediction.

<strong>ML vs Rule-Based Systems</strong>

Rule-Based systems involves manually setting rules and thresholds for prediction and classification however with ever changing data, these systems fails terribly or required a lot of resources and time to adjust to changes.

this is where Machine learning come to the save. Machine learning model is trained to find underlying patterns in given data and develop its own thresholds and rules with the help of probability and historical data points.

let's take an example with spam filter. 

using a rule-based system we can define indicators in the email which can define email as a spam or not such as sender email and email-body contents however sneeky email-sender would quickly bypass this by simply avoiding these rules.

with machine learning we can use our training dataset and find out underlying patterns in emails which would suggest if email is spam or not without necessarily knowing them beforehand.

<strong>types of Machine learning</strong>

there are two types:

- Supervised machine learning: they rely on data for which a known target exists (often referred to as labels). these predict output based on training datasets with input-output pairs.

- unsupervised machine learning/reinforced learning:discover hidden patterns or data groupings without the need of having input-output pairs in training dataset. these fall outside scope of this class.

<strong>supervised machine learning</strong>

 As said, in supervised machine learning, features are associated with labels. model is trained to associate features with particular labels. this can be done by threshold cutoffs or discrete value association.

There is features and targets.

features matrix is rows as observations and columns as attributes
target matrix: usually a verctor with information we want to predict

<strong>Types of Supervised ML problems</strong>

- Regression: the output is a number (car's prize)
- Classification: the output is a category (spam example).
- Ranking: the output is the big scores associated with certain items. items are ranked according to their measuring attributes(recommender systems)

<strong>CRISP-DM</strong>

The CRoss Industry Standard Process for Data Mining (CRISP-DM) is a methodology for organizing ML projects invented by IBM.

- <em>Business understanding</em>:do we need ML for the project. are the benefits outweighting costs and uncertainty manageble?
Data understanding: Analyze available data sources, and decide if more data is required and transformations that would be needed.
- <em>Data preparation</em>: Clean data and remove noise applying pipelines, and the data should be converted to a tabular format, so we can put it into ML.
- <em>Modeling</em>: training data on different models and choose the best one. consider if you would need additional features, data or remove redundant features.
- <em>Evaluation</em>: Measure how well the model is performing and if it solves the business problem.
- <em>Deployment</em>: Roll out to production to all the users. The evaluation and deployment often happen together

<strong>Environment</strong>
The main programming language used for the course is Python. It is a simple, yet robust language when it comes to handling data. at the time of taking, I used Python 3.9. We used Anaconda python distribution because of benefits of having important data science library we would need; these includes:

- <em>NumPy</em>: python library for scientific computing expecially with arrays.
- <em>Pandas</em>: python library for handling tabular data
- <em>Scikit-Learn</em>:python library for machine-learning models
- <em>Matplotlib and Seaborn</em>: python library for data visualization
- <em>Jupyter notebooks</em>: web application for sharing computing documents with input scripting and plain-text capabilities

## Week 2 - Regression

<strong>Introduction</strong>

Regression is a supervised machine learning technique. it is used for investigating relation between independent variables and dependent variables. it predicts a dependent variable using a set of indepent variables.

in this week, we worked on a project for predicting car prices using regression. dataset used was from this [kaggle competition](https://www.kaggle.com/datasets/CooperUnion/cardataset).

<em>project plan:</em>

- [Prepare data and Exploratory data analysis (EDA)](#EDA)

- [Validation Framework](#validate)

- [linear regression](#regression)

- [Evaluating the model](#rmse)

- [Feature engineering](#featureEng)

- [Regularization](#Regularization)

- [Using the model](#model)


<strong>Prepare data and Exploratory data analysis (EDA)</strong> <a name="EDA"></a>

```python
#import necessary libraries for EDA
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
```
read data and preview the first elements in the table

```python
df = pd.read_csv('data.csv')
df.head()
```
<div style="overflow-x: scroll;">
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
<table  markdown="block" border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Make</th>
      <th>Model</th>
      <th>Year</th>
      <th>Engine Fuel Type</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Transmission Type</th>
      <th>Driven_Wheels</th>
      <th>Number of Doors</th>
      <th>Market Category</th>
      <th>Vehicle Size</th>
      <th>Vehicle Style</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BMW</td>
      <td>1 Series M</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>335.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Factory Tuner,Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>26</td>
      <td>19</td>
      <td>3916</td>
      <td>46135</td>
    </tr>
    <tr>
      <th>1</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>19</td>
      <td>3916</td>
      <td>40650</td>
    </tr>
    <tr>
      <th>2</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>300.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,High-Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>20</td>
      <td>3916</td>
      <td>36350</td>
    </tr>
    <tr>
      <th>3</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury,Performance</td>
      <td>Compact</td>
      <td>Coupe</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>29450</td>
    </tr>
    <tr>
      <th>4</th>
      <td>BMW</td>
      <td>1 Series</td>
      <td>2011</td>
      <td>premium unleaded (required)</td>
      <td>230.0</td>
      <td>6.0</td>
      <td>MANUAL</td>
      <td>rear wheel drive</td>
      <td>2.0</td>
      <td>Luxury</td>
      <td>Compact</td>
      <td>Convertible</td>
      <td>28</td>
      <td>18</td>
      <td>3916</td>
      <td>34500</td>
    </tr>
  </tbody>
</table>
</div>



Some table properties


```python
#get the destribution statistics for numerical columns/attributes
df.describe()
```




<div style="overflow-x: scroll;">
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
      <th>Year</th>
      <th>Engine HP</th>
      <th>Engine Cylinders</th>
      <th>Number of Doors</th>
      <th>highway MPG</th>
      <th>city mpg</th>
      <th>Popularity</th>
      <th>MSRP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11914.000000</td>
      <td>11845.00000</td>
      <td>11884.000000</td>
      <td>11908.000000</td>
      <td>11914.000000</td>
      <td>11914.000000</td>
      <td>11914.000000</td>
      <td>1.191400e+04</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2010.384338</td>
      <td>249.38607</td>
      <td>5.628829</td>
      <td>3.436093</td>
      <td>26.637485</td>
      <td>19.733255</td>
      <td>1554.911197</td>
      <td>4.059474e+04</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.579740</td>
      <td>109.19187</td>
      <td>1.780559</td>
      <td>0.881315</td>
      <td>8.863001</td>
      <td>8.987798</td>
      <td>1441.855347</td>
      <td>6.010910e+04</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1990.000000</td>
      <td>55.00000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>12.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>2.000000e+03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2007.000000</td>
      <td>170.00000</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>22.000000</td>
      <td>16.000000</td>
      <td>549.000000</td>
      <td>2.100000e+04</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2015.000000</td>
      <td>227.00000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>26.000000</td>
      <td>18.000000</td>
      <td>1385.000000</td>
      <td>2.999500e+04</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2016.000000</td>
      <td>300.00000</td>
      <td>6.000000</td>
      <td>4.000000</td>
      <td>30.000000</td>
      <td>22.000000</td>
      <td>2009.000000</td>
      <td>4.223125e+04</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2017.000000</td>
      <td>1001.00000</td>
      <td>16.000000</td>
      <td>4.000000</td>
      <td>354.000000</td>
      <td>137.000000</td>
      <td>5657.000000</td>
      <td>2.065902e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
#number of entries/data rows
df.shape[0]
```




    11914




```python
#number of attributes/columns
df.shape[1]
```




    16




```python
#formating non-numeral columns to remove spaces
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for attr in string_columns:
    df[attr] = df[attr].str.lower().str.replace(' ', '_')
```

```python
plt.figure(figsize=(7, 5))

sns.histplot(df.msrp, bins=40, alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')

plt.show()
```


    
![Distribution of prices](/gallery/output_9_0.png)
    


distribution of prices, focusing on prices with lower frequencies



```python
plt.figure(figsize=(6, 4))

sns.histplot(df.msrp[df.msrp < 100000], bins=40, alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Price')
plt.title('Distribution of prices')

plt.show()
```


    
![lower Distribution of prices](/gallery/output_11_0.png)
    


check for columns with null attributes


```python
df.isnull().sum()
```




    make                    0
    model                   0
    year                    0
    engine_fuel_type        3
    engine_hp              69
    engine_cylinders       30
    transmission_type       0
    driven_wheels           0
    number_of_doors         6
    market_category      3742
    vehicle_size            0
    vehicle_style           0
    highway_mpg             0
    city_mpg                0
    popularity              0
    msrp                    0
    dtype: int64



Long-tail distributions usually confuse the ML models, so the recommendation is to transform the target variable distribution to a normal one whenever possible. we use log transformation since our data have a long tail suggesting logarithmic distribution


```python
log_price = np.log1p(df.msrp)

plt.figure(figsize=(6, 4))

sns.histplot(log_price, bins=40, alpha=1)
plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Distribution of prices after log tranformation')

plt.show()
```


    
![logarithmic distribution of prices](/gallery/output_15_0.png)
    

<strong>Validation Framework</strong> <a name="validate"></a>

The dataset is split into three parts: training, validation, and test. For each partition, we obtain feature matrices (X) and y vectors of targets.
The size of partitions is calculated, records are shuffled to guarantee that values of the three partitions contain non-sequential records of the dataset.


```python
#to allow reproducibility when the code is rerun the second time
np.random.seed(2)

n=len(df) #number of rows

#performing the split according to ratios
n_val = int(0.2 * n)
n_test = int(0.2 * n)
n_train = n - (n_val + n_test)

```


```python
idx = np.arange(n) #array of numbers from 0 to n

```


```python
#shuffle the number to ensure randomness
np.random.shuffle(idx)

#shuffle table
df_shuffled = df.iloc[idx]
```


```python
#partition
df_train = df_shuffled.iloc[:n_train].copy()
df_val = df_shuffled.iloc[n_train:n_train+n_val].copy()
df_test = df_shuffled.iloc[n_train+n_val:].copy()
```


```python
y_train_orig = df_train.msrp.values
y_val_orig = df_val.msrp.values
y_test_orig = df_test.msrp.values

#log transformation
y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

```

<strong>Linear Regression</strong> <a name="regression"></a>

```python
def train_linear_regression(X,y):
    ones=np.ones(X.shape[0])
    X=np.column_stack([ones,X])
    XTX=X.T.dot(X)
    XTX_inv=np.linalg.inv(XTX)
    w=XTX_inv.dot(X.T).dot(y)
    return w[0],w[1:]
```

######prediction


```python
base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']

def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)
print(w_0,w)
```

    7.927257388070001 [ 9.70589522e-03 -1.59103494e-01  1.43792133e-02  1.49441072e-02
     -9.06908672e-06]
    


```python
#prot for measuring descrepancy
y_pred = w_0 + X_train.dot(w)
plt.figure(figsize=(6, 4))
sns.histplot(y_train, label='target', color='blue', alpha=0.8, bins=40)
sns.histplot(y_pred, label='prediction', color='yellow', alpha=0.8, bins=40)

plt.legend()

plt.ylabel('Frequency')
plt.xlabel('Log(Price + 1)')
plt.title('Predictions vs actual distribution')

plt.show()

```


    
![Predictions vs actual distribution](/gallery/output_27_0.png)
    

<strong>RMSE Evaluation</strong> <a name="rmse"></a>

uses root mean square error to measure efficiency of model. he RMSE represents the differences between predicted values and observed values. the lower the RSME the better the model.

```python
def rmse(y, y_pred):
    error = y_pred - y
    mse = (error ** 2).mean()
    return np.sqrt(mse)
rmse(y_train, y_pred)
```

    0.7554192603920132


```python
#from validation set
X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
rmse(y_val, y_pred)
```

    0.7616530991301594

<strong>feature engineering</strong> <a name="featureEng"></a>

feature engineering is manipulating dataset such as addition, deletion, combination, mutation to improve machine learning model training, leading to better performance and greater accuracy.our goal is to have as smaller RSME as possible

```python
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```


```python
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation', rmse(y_val, y_pred))
```

    train 0.5175055465840046
    validation 0.5172055461058324
    

```python
df['make'].value_counts().head(5)
```




    chevrolet     1123
    ford           881
    volkswagen     809
    toyota         746
    dodge          626
    Name: make, dtype: int64




```python
top=df['make'].value_counts().head(5)
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')

    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)
        
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```


```python
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)
print('train:', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred))
```

    train: 0.5058876515487503
    validation: 0.5076038849555671
    


```python
def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')
    
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)', 
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)
        
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```


```python
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)
print('train:', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred))
```

    train: 0.4745380510924003
    validation: 0.46858791946591755
    


```python
df['driven_wheels'].value_counts()
```




    front_wheel_drive    4787
    rear_wheel_drive     3371
    all_wheel_drive      2353
    four_wheel_drive     1403
    Name: driven_wheels, dtype: int64




```python

df['market_category'].value_counts().head(5)
```




    crossover             1110
    flex_fuel              872
    luxury                 855
    luxury,performance     673
    hatchback              641
    Name: market_category, dtype: int64




```python
df['vehicle_size'].value_counts().head(5)
```




    compact    4764
    midsize    4373
    large      2777
    Name: vehicle_size, dtype: int64




```python
df['vehicle_size'].value_counts().head(5)
```




    compact    4764
    midsize    4373
    large      2777
    Name: vehicle_size, dtype: int64




```python

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')
    
    for v in [2, 3, 4]:
        feature = 'num_doors_%s' % v
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)

    for v in ['chevrolet', 'ford', 'volkswagen', 'toyota', 'dodge']:
        feature = 'is_make_%s' % v
        df[feature] = (df['make'] == v).astype(int)
        features.append(feature)

    for v in ['regular_unleaded', 'premium_unleaded_(required)', 
              'premium_unleaded_(recommended)', 'flex-fuel_(unleaded/e85)']:
        feature = 'is_type_%s' % v
        df[feature] = (df['engine_fuel_type'] == v).astype(int)
        features.append(feature)

    for v in ['automatic', 'manual', 'automated_manual']:
        feature = 'is_transmission_%s' % v
        df[feature] = (df['transmission_type'] == v).astype(int)
        features.append(feature)

    for v in ['front_wheel_drive', 'rear_wheel_drive', 'all_wheel_drive', 'four_wheel_drive']:
        feature = 'is_driven_wheens_%s' % v
        df[feature] = (df['driven_wheels'] == v).astype(int)
        features.append(feature)

    for v in ['crossover', 'flex_fuel', 'luxury', 'luxury,performance', 'hatchback']:
        feature = 'is_mc_%s' % v
        df[feature] = (df['market_category'] == v).astype(int)
        features.append(feature)

    for v in ['compact', 'midsize', 'large']:
        feature = 'is_size_%s' % v
        df[feature] = (df['vehicle_size'] == v).astype(int)
        features.append(feature)

    for v in ['sedan', '4dr_suv', 'coupe', 'convertible', '4dr_hatchback']:
        feature = 'is_style_%s' % v
        df[feature] = (df['vehicle_style'] == v).astype(int)
        features.append(feature)
   
    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X
```


```python
X_train = prepare_X(df_train)
w_0, w = train_linear_regression(X_train, y_train)

y_pred = w_0 + X_train.dot(w)
print('train:', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred))
```

    train: 98.83799994732918
    validation: 90.75198555498365
    


```python
w_0
```




    -1.1135946518113574e+16


<strong>Regularization</strong> <a name="regularization"></a>

```python
def train_linear_regression_reg(X, y, r=0.0):
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])

    XTX = X.T.dot(X)
    reg = r * np.eye(XTX.shape[0])
    #print(reg)
    XTX = XTX + reg

    XTX_inv = np.linalg.inv(XTX)
    w = XTX_inv.dot(X.T).dot(y)
    
    return w[0], w[1:]
X_train = prepare_X(df_train)
for r in [0, 0.001, 0.01, 0.1, 1, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    print('%5s, %.2f, %.2f, %.2f' % (r, w_0, w[13], w[21]))
```

        0, -11135946518113574.00, 12.19, 11135946518113572.00
    0.001, 7.20, -0.10, 1.81
     0.01, 7.18, -0.10, 1.81
      0.1, 7.05, -0.10, 1.78
        1, 6.22, -0.10, 1.56
       10, 4.39, -0.09, 1.08
    


```python
X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0)

y_pred = w_0 + X_train.dot(w)
print('train', rmse(y_train, y_pred))

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('val', rmse(y_val, y_pred))
```

    train 98.83799994732918
    val 90.75198555498365
    


```python
X_train = prepare_X(df_train)
X_val = prepare_X(df_val)

for r in [0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]:
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    print('%6s' %r, rmse(y_val, y_pred))
```

     1e-06 0.46022485228775944
    0.0001 0.4602254918041578
     0.001 0.4602267630259776
      0.01 0.460239496285643
       0.1 0.4603700695839839
         1 0.4618298042649753
         5 0.4684079627532594
        10 0.4757248100693656
    


```python
X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=0.01)

X_val = prepare_X(df_val)
y_pred = w_0 + X_val.dot(w)
print('validation:', rmse(y_val, y_pred))

X_test = prepare_X(df_test)
y_pred = w_0 + X_test.dot(w)
print('test:', rmse(y_test, y_pred))
```

    validation: 0.460239496285643
    test: 0.4571813679219644
    
<strong>Using the model</strong> <a name="model"></a>

```python
i = 2
ad = df_test.iloc[i].to_dict()
ad
```




    {'make': 'toyota',
     'model': 'venza',
     'year': 2013,
     'engine_fuel_type': 'regular_unleaded',
     'engine_hp': 268.0,
     'engine_cylinders': 6.0,
     'transmission_type': 'automatic',
     'driven_wheels': 'all_wheel_drive',
     'number_of_doors': 4.0,
     'market_category': 'crossover,performance',
     'vehicle_size': 'midsize',
     'vehicle_style': 'wagon',
     'highway_mpg': 25,
     'city_mpg': 18,
     'popularity': 2031}


X_test = prepare_X(pd.DataFrame([ad]))[0]
y_pred = w_0 + X_test.dot(w)
suggestion = np.expm1(y_pred) #this is to ensure greater precision than exp(x) - 1 for small values of x
suggestion

