src: https://github.com/eugeneyan/ml-design-docs

# HDB Resale Price analysis with Ensemble learning techniques, by Joshua0611

## 1. Overview

<!--
A summary of the doc's purpose, problem, solution, and desired outcome, usually in 3-5 sentences.
-->

The HDB Resale Flat price regression analysis using Ensemble machine learning techniques (henceforth Project) is a project that aims to utilise Ensemble learning techniques to conduct a regression analysis of HDB resale flat prices (henceforth Prices) across a variety of feature types, including numerical, categorical, and time series data. 

The project aims to produce meaningful insights regarding the links between aforementioned features and HDB Resale Prices, through the use of an Ensemble machine learning model. 

This document therefore seeks to discuss the motivation behind the conceptualisation of the Project, as well as the process through which it was realised, and the key decisions made at critical points during its conceptualisation.
<br></br><br></br>

## 2. Motivation

<!--
Why the problem is important to solve, and why now.
-->

The Project was motivated by my inability to find any proper and/or rigorous analysis regarding the relationships between HDB flat features and Prices online. I therefore sought to create this project as a means to help those seeking to buy or sell HDB Resale flats make better decisions regarding the valuations of the flats under consideration, leveraging on AI and machine learning.

This project, my first machine learning project, was also motivated by the large variety of feature types within the data utilised, which I felt would allow for a good challenge. 
<br></br><br></br>

## 3. Success metrics

<!--
Usually framed as business goals, such as increased customer engagement (e.g., CTR, DAU), revenue, or reduced cost.
-->

The key success metric for this project is the creation of an Ensemble learning model, which after training, predicts HDB Resale Flat prices within $50 000 of the actual price of the flat, more than 85% of the time. 

A desired outcome of this project was also the generation of a report detailing the links between HDB Resale flat features and Prices, which is discussed in depth below.
<br></br><br></br>

## 4. Requirements & Constraints

<!--
Functional requirements are those that should be met to ship the project. They should be described in terms of the customer perspective and benefit. (See this for more details.)

Non-functional/technical requirements are those that define system quality and how the system should be implemented. These include performance (throughput, latency, error rates), cost (infra cost, ops effort), security, data privacy, etc.

Constraints can come in the form of non-functional requirements (e.g., cost below $x a month, p99 latency < yms)
-->

### The Functional and Technical requirements of the Project are as follows:

- The generation of a report exploring and detailing:
    - The relationship between each feature and Price.
    - The relationship between multiple features and Prices, for features from the same base pipeline.
    - The relative importance of each feature in explaining Price.
    - The relative importance of each group of features corresponding to each pipeline in explaining Price.
- The generation of a trained Ensemble supervised learning model that fulfils the relevant success metric as outlined above.

### The constraints and restrictions of the Project are as follows:
- The model utilised must require under 12gbs of RAM to train and utilise.

## 4.1 What's in-scope & out-of-scope?

### The tasks in scope are as follows:

- Report related tasks:
    - See the objectives of the report as mentioned above.

- Programming tasks:
    - Conduct of sufficient data analysis purely for the sake of model architecture designing and hyperparameter tuning, and for use in the report.

### Some tasks out of scope are as follows:

- Report related tasks:
    - Analysis of relationship between features beyond analysis of correlation.
    - Explanation of standard mathematical formulae (such as Mean Squared Error (henceforth MSE)).
    - Explanation of the lower level workings of the model used and project methodologies beyond what is sufficient for the fulfilment of the objectives of the report, as discussed above. 

- Programming related tasks:
    - Generation of interactive, dynamic or animated visuals.
    - Front end integration of the model.
    - User interface design related tasks.
<br></br><br></br>

## 5. Methodology

## 5.1. Problem statement

<!--
How will you frame the problem? For example, fraud detection can be framed as an unsupervised (outlier detection, graph cluster) or supervised problem (e.g., classification).
-->

The problem to be solved is primarily supervised in nature, and revolves around the use of supervised learning and ensemble modelling techniques to explore the relationship between the features of a HDB Resale Flat and its price.

## 5.2. Data
<!--
What data will you use to train your model? What input data is needed during serving?
-->

### The data used for this analysis was sourced from XXX. It was extracted as a set of .csv files, containing 901116 entries and the following columns:

- town: The HDB town the flat was from (eg. ANG MO KIO).

- flat_type: The type of the flat (eg. 1 ROOM).

- block: The block number of the flat, formatted as integers of up to 3 digits (eg. 334), some of which have a single alphabetical character at the end (eg. 844A).
    - Cardinality: 2666
    - Min: 1, 9A
    - Max: 980 ,997C

- street_name: The street name the flat was on (eg. ANG MO KIO ST 5).
The data has a cardinality of 582

- flat_model:  The model of the flat (eg. DBSS).
    - The data has a cardinality of 34, but only about 25 “true” unique values, due to issues regarding the formatting of string values (eg. “Multi generational” vs “MULTI GENERATIONAL”)

- storey_range: The storey range the flat was in, formatted as “lower range” TO “upper range” (eg. “04 TO 07”).
    - The data range contains ranges with a minimum minimum value of 01, to a maximum maximum value of 51

- floor_area_sqm: The floor area of the flat, in square metres (eg. 102).
    - Mean: 95.706282
    - Standard deviation: 25.881837
    - Min: 28
    - Max: 307

- month: The month (and year) the sale of the flat was conducted, formatted as “YYYY-MM”.
    - The data range is from Jan 1990 to June 2023

- lease_commence_date: The year of commencement of the lease of the flat, formatted as YYYY (eg. 1969).
The data range is from 1966 to 2019

- resale_price: The nominal resale price of the flat, at the point of transaction, correct to the nearest Singapore Dollar (eg. 442244). 
    - Mean: 313 119.5
    - Standard deviation: 163969.4
    - Min: 5000
    - Max: 1 418 000
    
### The columns used for model training were:
Categorical: 
- town 
- flat_type (renamed to "type")
- flat_model (renamed to "model")

Numerical:
- floor_area_sqm (renamed to "size")
- storey_range (renamed to "storey"), with ranges replaced with average value within the range (eg. 01 TO 03 was replaced with 2)
- lease_left, synthetic feature generated by extracting the time between "month" and "lease_start", and deducting it from 99, rounded to the nearest year

Time series:
- lease_start
- month (renamed to "sale_date")

### Data to be used for predicting with model generated:
The data to be used for predicting prices given new data should be formatted similarly as the above.

## 5.3. Techniques

<!--
What machine learning techniques will you use? How will you clean and prepare the data (e.g., excluding outliers) and create features?
-->

### Overall model architecture
The model used in this project is a supervised learning one, that relies on ensemble learning techniques to achieve its objectives.

This was achieved by first preprocessing the data in general, then splitting it for feeding into multiple pipelines.

### Preprocessing data and feature engineering
In general, all numerical data (Price included) was normalised. No truncation or any other form of exclusion or censoring was conducted for the numerical data as this was found to cause the models used to perform more poorly in general (implying, unexpectedly, a method to the madness that is the determination of million dollar HDB flats). 

Categorical features that occurred less than 5% the average number of times each category occurred, with respect to that feature, were truncated, following this. Next, all relevant categorical features underwent one hot encoding. Lastly, the relevant categorical features were then crossed, in a manner that allowed for the presence of "partial" segments (feature crosses containing features from a subset of all categorical features) and "full" segments (feature crosses containing features from all categorical features).

Numerical features were further crossed and polynomialised to a maximum degree equal to the number of numerical features.

Time series features were aggregated into Pandas Series objects, with the indices corresponding to the datetimes that dataset contained, under that feature, and the values corresponding to the average price of all entries corresponding to that datetime. However, this was done after the time series data was split into train and test sets, for both train and test sets of each time series feature.

### Splitting data

All features' data was split roughly or exactly 80-20, with the former set of data used for model training and the latter for model evaluation. This was done randomly for the time series and numerical features.

However, categorical features were split in a stratified manner, feature-wise, to ensure that both train and test splits contained data containing data entries which had all categories in each feature. 

### Model selection and hyperparameter tuning
Models were selected to maximise performance, but also be white box in themselves. 

The meta, categorical, and numerical pipelines' models (mainly linear or ridge regression models) were selected to minimise their R2 scores. The time series pipelines' models (mainly ARIMA models) were selected to minimise their Akaike Information Criterion (AIC) scores, with the constraint that models were selected to ensure the non-occurrence of unit roots in them.

## 5.4. Experimentation & Validation

<!--
How will you validate your approach offline? What offline evaluation metrics will you use?

If you're A/B testing, how will you assign treatment and control (e.g., customer vs. session-based) and what metrics will you measure? What are the success and guardrail metrics?
-->

The base and meta models' hyperparameters were tuned using their corresponding scoring metrics, but they were additionally evaluated based on the proportion of predictions they made that were within 5%, 10%, 15%, 20%, 25%, and 30%, of the true values of the data they predicted on.
<br></br><br></br>

## 6. Implementation

## 6.0 High level design

<!--
Start by providing a big-picture view. System-context diagrams and data-flow diagrams work well.
-->

### Overall model architecture

The overall architecture of the model used involved the feeding of generally preprocessed data split appropriately, into 4 base pipelines. Each pipeline would preprocess its data, feed it into its own model, and produce predictions for the prices of a flats corresponding to the data fed into it. The predictions produced from these pipelines would then be fed into a meta pipeline, which would feed these predictions into a stacking regressor, before generating the final prediction. This is illustrated below.

![Diagram of overall, high level model architecture](Docs%20diagrams/High%20level.png "Diagram of overall, high level model architecture")


### Categorical pipeline
In the case of the categorical pipeline, splitting of the data in a stratified manner (as discussed earlier) was done before preprocessing. This was to ensure that train and test splits of the data contained instances of all categories present in each feature. Following this, the data was fed into a ridge regression model for prediction and analysis. This is illustrated below.

![Diagram of categorical pipeline](Docs%20diagrams/Cat%20base.png "Diagram of categorical pipeline") 

### Numerical pipeline
In the case of the numerical pipeline, the data was preprocessed, split, and fed into a linear regression model for prediction and analysis. This is illustrated below.

![Diagram of numerical pipeline](Docs diagrams\Num base.png)  

### Time series pipelines
In the case of the time series pipelines, similarly to the numerical pipeline, the data was preprocessed, then split, then fed into ARIMA models of appropriate order for prediction, and analysis. This is illustrated below.

![Diagram of lease start pipeline](Docs%20diagrams/LS%20base.png "Diagram of lease start pipeline")
![Diagram of sale date pipeline](Docs%20diagrams/SD%20base.png "Diagram of sale date pipeline")

## 6.1. Cost
<!--
How much will it cost to build and operate your system? Share estimated monthly costs (e.g., EC2 instances, Lambda, etc.)
→

The only cost of constructing this project is my time. 

The project took me about 300 hours to complete. It will probably take under 30 minutes to download the necessary data and run the codes in this project. 

### 6.2. Integration points
<!--
How will your system integrate with upstream data and downstream users?
-->

Due to the nature of this project, no upstream or downstream system integrations were planned or constructed.

## 6.2. Risks & Uncertainties
<!--
Risks are the known unknowns; uncertainties are the unknown unknowns. What worries you and you would like others to review?
-->

There are a number of risks regarding the rigour of this project, its ensemble model, and the analysis, despite my best efforts to mitigate or otherwise eliminate them.

These include:

1. The non-inclusion of flats' street names into the analysis, due to that feature's high cardinality and my lack of computing resources. This would probably have led to a lack of inclusion of additional information such as proximity to key transport nodes into the analysis, due to its encoding within the information regarding the streets the flats were on when sold.

2. The use of aggregated data for time series analysis, which I utilised due to the lack of suitable segments with which I could (in my opinion) satisfactorily analyse price changes relative to the appropriate time series features, controlling for other exogenous factors. This could have led to the mis-representativeness of the relationship between the time series feature values and HDB resale price, due to varying prevalences of segments (eg. there could have been a greater proportion of larger flats (which tends to correspond to higher prices) built as time went on, leading to higher average prices over time)  observed over time.

3. The use of ARIMA models rather than SARIMA or even SARIMAX models, which may have better modelled the Time series data than the ARIMA models used.

Furthermore, there are likely to be more risks and uncertainties that I could have not foreseen, due to my relative inexperience in the conduct of such a project (it is my first machine learning project).
<br></br><br></br>

## 7. Appendix

## 7.1. Alternatives considered

<!--
What alternatives did you consider and exclude? List pros and cons of each alternative and the rationale for your decision.
-->

Below are a list of key decisions made, and why they were made in light of potential alternative decisions that could have been made in their place:

### General
- Preprocessing and splitting before feeding into base models

    - Not including the block number of the flats. This was due to my belief that block numbers do not encapsulate any meaningful information about the flat.

### Categorical pipeline

- Preprocessing, splitting, and selection of data.

    - During truncation of categorical features, not combining "outlier" classes into an "other" class. This was due to my belief that such a class would be meaningless given the variety of actual classes this class would be composed of.
    
    - Creation of "lease_left", a synthetic feature encoding the time between lease_start and sale_date. Inasmuch as this feature is clearly collinear with lease_start and sale_date, each feature was used in the training of a different base model, which I felt would mitigate any collinearity-related issues with model training.

    - Creating interaction feature crosses containing categories from all categorical features but also feature crosses that contained categories from only a subset of all categorical features. I did this to allow for superior segment analysis, and to allow for the overall pipeline to be more easily adapted to predict on data that does not have all the categorical features.
    
- Training and selection of model and hyperparameters, and model evaluation.

    - Use of R2 score instead of other metrics such as MAE or MSE, for the selection of the relevant models, and the tuning of their hyperparameters. This was to prioritise explanatory power of the models in their selection and hyperparameter tuning, to aid in the aforementioned analysis [1].

    - Use of a Ridge Regression model instead of models more "traditionally" associated with categorical data (to me), such as KNNRegressors or Support Vector Regressors (SVRs). This was due to (somewhat surprisingly to me) the Ridge Regression model performing better than the aforementioned alternative model types, and its relative interpretability.

### Numerical pipeline

- Preprocessing, splitting, and selection of data

    - Lack of truncation or censoring of numerical features. This was as I found that doing so led to poorer numerical model performance, after adjusting for hyperparameters as needed.

- Training and selection of model and hyperparameters, and model evaluation.

    - Use of 5-fold cross validation R2 scores as a model evaluation metric for model selection and hyperparameter tuning, instead of other potential metrics. Same as for the categorical pipeline (see above).

### Time series pipelines

- Preprocessing, splitting, and selection of data.

    - Use of average prices for all possible segments corresponding to each feature value, instead of the average prices of fixed segments across the time series feature range. This was as I was unable to find any segment that was prevalent enough across the time series feature range to be used for this purpose. 
        - One technique considered to mitigate this was the aggregation of the data corresponding to the relevant feature values into fixed intervals. However, this was unsuccessful in solving the issue at hand.

- Training and selection of model and hyperparameters, and model evaluation.

    - Reliance on ACF and PACF plots rather than decomposition plots in hyperparameter selection. This was as I felt that the ACF and PACF plots were sufficient to aid in the selection of hyperparameters. However, it must be noted that the decomposition graphs were indeed generated, but not heavily utilised in the selection of the model hyperparameters.

## 7.2. Experiment Results

<!--
Share any results of offline experiments that you conducted.
-->

See the report on the findings of this project here. [HDB Resale Price analysis.docx](HDB%20Resale%20Price%20analysis.docx)

## 7.3. Performance benchmarks

<!--
Share any performance benchmarks you ran (e.g., throughput vs. latency vs. instance size/count).
-->

The performance of the pipelines and models based on my own experiments are as shown below:

![Prevalence of model percentage errors for all models](Docs%20diagrams/Percentage%20errors%20across%20models.png "Prevalence of model percentage errors for all models")

![Prevalence of absolute model errors for meta model](Docs%20diagrams/Model%20performance%20graph.png "Prevalence of absolute model errors for meta model")

## 7.4. References

<!--
Add references that you might have consulted for your methodology.
--> 
### Referenced material:

1.	Hale, J. (2020, Aug 20). Which Evaluation Metric Should You Use in Machine Learning Regression Problems? Retrieved Jun 25, 2023, from https://towardsdatascience.com/which-evaluation-metric-should-you-use-in-machine-learning-regression-problems-20cdaef258e

### Material consulted but not referenced above:

2.	Nau, R. (2020, Aug 18). ARIMA models for time series forecasting. Statistical forecasting: notes on regression and time series analysis. Retrieved Jul 12, 2023, from https://people.duke.edu/~rnau/411arim3.htm

3.	Brownlee, J. (2020, Aug 15). A Gentle Introduction to the Box-Jenkins Method for Time Series Forecasting. Retrieved Jul 12, 2023, from https://machinelearningmastery.com/gentle-introduction-box-jenkins-method-time-series-forecasting/










