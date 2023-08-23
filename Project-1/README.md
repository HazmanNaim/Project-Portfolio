# Predicting House Sales Price in King Country
Duration: 5 Days
## Purpose:
This project aims to develop a machine learning model to predict house sales prices in King County. The goal is to provide accurate price estimates that empower homebuyers, sellers, and real estate professionals to make informed decisions. By analyzing property attributes and historical sales data, the model aims to identify key factors influencing house prices, offer insights into market trends, and contribute to fair and well-informed real estate transactions.

## Methodology:
I employed a comprehensive dataset encompassing house sale prices in King County from May 2014 to May 2015, sourced from Kaggle. The dataset can be accessed through this ([Kaggle Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-wwwcourseraorg-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01)) link. Subsequently, I conducted exploratory data analysis, involving data refinement and the exploration of correlations and associations among the dataset features. The last phase involved constructing machine learning models based on the processed datasets. To achieve this, I will be using various machine learning techniques, including linear regression, multiple linear regression, support vector regression, and random forest regressor provided by scikit-learn.

Throughout this notebook, I will also perform necessary pre-processing on the dataset. In addition, I will use 5-fold cross-validation to ensure reliable model assessment and make use of hyperparameter tuning to fine-tune the models. At the conclusion of this notebook, I will provide visualizations that showcase the performance of these models. These visualizations will include residual plots, scatter plots, and distribution plots.

## Tools/Tech
![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

## Challenges Encountered
1. The project encountered challenges primarily stemming from hardware limitations. The use of a low-specification computer significantly impeded progress. Tasks such as model tuning, which necessitate numerous iterations, were notably delayed due to hardware constraints. For instance, the time required to fine-tune the model was substantially extended, hindering the overall workflow.

## Outcomes
**Results**

The project involved the development of 17 distinct machine learning models utilizing various algorithms. These models included Ordinary Least Squares (OLS), Ridge Regression, Lasso, Multiple Linear Regression, Support Vector Regression and Random Forest. 

A summary of the models' performance metrics is provided below:

| Model                                 | Model Alias | Detail                                             | R2 Score (Train) | Adjusted R2 Score (Train) | RMSE (Test) | R2 Score (Test) | Adjusted R2 Score (Test) | Cross-Validated RMSE | Cross-Validated R2 Score |
|---------------------------------------|-------------|----------------------------------------------------|------------------|--------------------------|-------------|----------------|--------------------------|----------------------|--------------------------|
| Random Forest Regressor-3              | rf3         | Features (All), No Pre-process                    | 0.982947         | 0.9829326370940544       | 142265.752505 | 0.873822       | 0.8737173081576125       | 129475.668059        | 0.874010                 |
| Random Forest Regressor-1              | rf1         | Features (All), Pre-process                        | 0.982732         | 0.9827052971543786       | 141516.332050 | 0.875148       | 0.8749573960126045       | 130135.364924        | 0.876313                 |
| Random Forest Regressor-4              | rf4         | min_samples_split = 2, min_samples_leaf = 3, ...  | 0.962571         | 0.962514                | 144843.083014 | 0.869209       | 0.869009                 | 131603.223959        | 0.870560                 |
| Random Forest Regressor-2              | rf2         | Features (corr>0.1), Pre-process                  | 0.973770         | 0.9737530277606906       | 172362.541909 | 0.814789       | 0.8146688414706987       | 158550.178204        | 0.810960                 |
| Ridge Regression-2                     | rm2         | alpha = 10, Features (All), Pre-process           | 0.708865         | 0.7084199704126287       | 223701.621594 | 0.688025       | 0.6875481168292614       | 200153.883927        | 0.701806                 |
| Lasso Regression-2                     | lm2         | alpha = 10, Features (All), Pre-process           | 0.709008         | 0.7085634007407488       | 223188.100530 | 0.689456       | 0.6889809760107755       | 200165.206551        | 0.701753                 |
| Ridge Regression-1                     | rm1         | alpha = 1, Features (All), Pre-process            | 0.709010         | 0.7085645789249735       | 223218.552226 | 0.689371       | 0.6888960995971063       | 200167.338096        | 0.701749                 |
| Lasso Regression-1                     | lm1         | alpha = 1, Features (All), Pre-process            | 0.709012         | 0.708566794560971        | 223166.633700 | 0.689516       | 0.689040802419646        | 200175.542399        | 0.701722                 |
| Multiple Linear Regression-4           | mlr4        | Features (All), Pre-process                      | 0.709047         | 0.7086019647440626       | 223105.241120 | 0.689686       | 0.6892118670748867       | 200182.208543        | 0.701702                 |
| Ridge Regression-3                     | rm3         | alpha = 100, Features (All), Pre-process          | 0.702932         | 0.7024778832184858       | 227737.911832 | 0.676666       | 0.6761711404073993       | 202259.404430        | 0.695569                 |
| Multiple Linear Regression-1           | mlr1        | Features (All), No Pre-process                   | 0.702288         | 0.7020393483390373       | 226408.020326 | 0.680431       | 0.6801644644849335       | 202505.041196        | 0.694683                 |
| Multiple Linear Regression-2           | mlr2        | Features (corr>0.1), Pre-process                 | 0.664286         | 0.6640684749887042       | 237056.183239 | 0.649665       | 0.6494376790289351       | 213961.544884        | 0.658974                 |
| Lasso Regression-3                     | lm3         | alpha = 10, Features (corr>0.1), Pre-process     | 0.664063         | 0.6638454495088593       | 237086.588330 | 0.649575       | 0.649347746229997        | 213993.931520        | 0.658864                 |
| Ridge Regression-4                     | rm4         | alpha = 10, Features (corr>0.1), Pre-process     | 0.663934         | 0.6637162689787687       | 237539.606721 | 0.648234       | 0.6480064330351382       | 214021.996054        | 0.658806                 |
| Multiple Linear Regression-3           | mlr3        | Features (corr>0.5), Pre-process                | 0.545430         | 0.5453454849315925       | 273432.506691 | 0.533897       | 0.533811116248798        | 248197.088246        | 0.541285                 |
| Ridge Regression-5                     | rm5         | alpha = 10, Features (corr>0.5), Pre-process     | 0.545430         | 0.5453454151699382       | 273434.753928 | 0.533890       | 0.5338034533601883       | 248197.425303        | 0.541284                 |
| Simple Linear Regression                | slr         | sqft_living                                       | 0.493845         | -                        | 287447.932468 | 0.484890       | -                        | 261378.268667        | 0.491086                 |
| Support Vector Regression-2            | svr2        | C=10, Features (All), Pre-process                 | -0.059397        | -0.061016957746032566    | 412873.613544 | -0.062712      | -0.06433694765921087    | 377676.896860        | -0.062106                |
| Support Vector Regression-1            | svr1        | C=1, Features (All), Pre-process                  | -0.059860        | -0.06148083335336984     | 412979.053020 | -0.063255      | -0.06488063684927806    | 377707.257310        | -0.062272                |

From the provided summary table about how well models predict house prices using 5-fold cross-validation, we can learn some useful things:

1. **Best Model Performance:** The model that stood out the most is 'Random Forest Regressor-3' (rf3). It achieved high cross-validated R2 scores and low cross-validated RMSE values on both the training and test sets. This suggests its effectiveness in capturing the underlying patterns in the data, and these results are backed by the use of cross-validation.

2. **Hyperparameter Tuning and Preprocessing Impact:** Notably, the 'Random Forest Regressor-4' (rf4) stands out. Despite adjustments made through hyperparameter tuning, its performance remains comparable to the default setting model ('rf3'). This emphasizes that while hyperparameter tuning and preprocessing can enhance model outcomes, there's a point at which additional tuning might not lead to substantial improvements.

3. **Comparing with Baseline Models:** When we look at the Simple Linear Regression (slr) model, we see that it didn't perform well in comparison to other models. This means that more complex models like Random Forest and other regression techniques (Ridge, Lasso, Multiple Linear Regression) are better suited for capturing the relationships in this dataset.

4. **Support Vector Regression Performance:** The Support Vector Regression (svr) models didn't show good performance. However, it's important to consider that this might be due to issues in how the models were developed rather than the inherent nature of the algorithm.

5. **Effect of Feature Preprocessing:** Comparing the 'Random Forest Regressor-3' model with 'Random Forest Regressor-1' and 'Random Forest Regressor-2', it's clear that feature preprocessing might not significantly improve results in this specific context. The models with feature preprocessing didn't outperform the default setting model ('rf3') by a large margin.

In the end, the results from the models help us understand how well they predict house prices using 5-fold cross-validation, what kinds of methods work best, and how much changing things before running the models really matters. While some models are a bit different, the big idea is that we should choose models that fit the data and find a balance between changing features, tuning, and making things more complicated.

## Further Enhancements
1. Given a longer time duration, it would be beneficial to delve into more advanced machine learning models beyond the ones already explored. Models such as Gradient Boosting, or Neural Networks could be studied to potentially yield even more accurate predictions.
2. Testing the models on entirely separate datasets, including those not used during training or evaluation, can provide a more realistic assessment of their predictive accuracy and generalization to new data.
3. To ensure that the models are trained with the most relevant features, a more comprehensive approach to feature selection could be implemented. For instance, conducting techniques like Recursive Feature Elimination could be considered. This process iteratively identifies and removes less influential features, ultimately enhancing the model's ability to capture meaningful patterns and relationships within the data.
