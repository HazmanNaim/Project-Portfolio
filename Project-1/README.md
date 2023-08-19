# Predicting House Sales Price in King Country
Duration: 5 Days
## Purpose:
This project aims to develop a machine learning model to predict house sales prices in King County. The goal is to provide accurate price estimates that empower homebuyers, sellers, and real estate professionals to make informed decisions. By analyzing property attributes and historical sales data, the model aims to identify key factors influencing house prices, offer insights into market trends, and contribute to fair and well-informed real estate transactions.

## Methodology:
I employed a comprehensive dataset encompassing house sale prices in King County from May 2014 to May 2015, sourced from Kaggle. The dataset can be accessed through this ([Kaggle Dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-wwwcourseraorg-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDA0101ENSkillsNetwork20235326-2022-01-01)) link. Subsequently, I conducted exploratory data analysis, involving data refinement and the exploration of correlations and associations among the dataset features. The last phase involved constructing machine learning models based on the processed datasets.

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

The project involved the development and evaluation of four distinct machine learning models utilizing various algorithms. These models included Ordinary Least Squares (OLS), Ridge Regression, Lasso, and Random Forest. Following the analysis, the Random Forest model emerged as the most accurate, exhibiting a commendable accuracy level with an R-squared score of 78%.

A summary of the models' performance metrics is provided below:

| Model             | Mean Absolute Error | Mean Squared Error | R2-Score  |
|-------------------|---------------------|--------------------|-----------|
| Linear Regression | 136078.645759       | 5.175599e+10       | 0.647553  |
| Ridge Regression  | 119699.873833       | 4.401399e+10       | 0.700274  |
| Lasso             | 135857.054133       | 5.170751e+10       | 0.647883  |
| Random Forest     | 91012.840637        | 3.226826e+10       | 0.780260  |

The Random Forest model demonstrated the highest accuracy, achieving a substantial R2-Score of 78.03%. This outcome underscores the effectiveness of the chosen model in predicting house sales prices in the King County context.

## Further Enhancements
1. Given a longer time duration, it would be beneficial to delve into more advanced machine learning models beyond the ones already explored. Models such as Gradient Boosting, or Neural Networks could be studied to potentially yield even more accurate predictions.
2. Testing the models on entirely separate datasets, including those not used during training or evaluation, can provide a more realistic assessment of their predictive accuracy and generalization to new data.
3. To ensure that the models are trained with the most relevant features, a more comprehensive approach to feature selection could be implemented. For instance, conducting techniques like Recursive Feature Elimination could be considered. This process iteratively identifies and removes less influential features, ultimately enhancing the model's ability to capture meaningful patterns and relationships within the data.
