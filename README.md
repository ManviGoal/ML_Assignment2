Machine Learning Assignment2

# Problem Statement
The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual earns more than 50K USD per year based on demographic and employment-related attributes.  

# Dataset Description
The dataset used is the Census Income (Adult) Dataset from the UCI Machine Learning Repository. It contains demographic features such as age, education, occupation, marital status, hours worked per week, etc. The target variable is income, classified as <=50K or >50K.  

# Models Used and Evaluation Metrics

Comparison Table  
Model , Accuracy, AUC, Precision, Recall, F1  
0 , Logistic Regression,  0.827883,  0.860793,   0.724623,  0.459821,  0.562622  
1 ,       Decision Tree,  0.813604,  0.752396,   0.608191,  0.634566,  0.621099  
2 ,                 KNN,  0.834024,  0.856943,   0.671117,  0.609056,  0.638582  
3 ,         Naive Bayes,  0.808076,  0.864383,   0.704370,  0.349490,  0.467178  
4 ,       Random Forest,  0.859665,  0.911135,   0.744395,  0.635204,  0.685478  
5 ,             XGBoost,  0.876248,  0.928644,   0.776087,  0.683036,  0.726594

# Model Performance Observations

ML Model : Observation  
Logistic Regression : Performs well with balanced precision and recall, suitable as a baseline model.  
Decision Tree : Easy to interpret but prone to overfitting.  
KNN : Performs moderately well but sensitive to feature scaling.  
Naive Bayes : Fast and simple, but assumes feature independence.  
Random Forest : Strong performance due to ensemble learning and reduced variance.  
XGBoost : Best overall performance with highest AUC and MCC scores.  
