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

