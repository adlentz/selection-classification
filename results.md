##Comparison of Gaussian Naive Bayes, Logistic Regression, and SVC##
**Feature Selection:**

KBest Results:
array([  7.44286583e-02,   1.29892853e+00,   1.17730452e+03, 3.04270309e-01])

RFE Results:
array([False,  True,  True, False], dtype=bool), array([3, 1, 1, 2])

Chi2 Results:
array([  5.22083508e-03,   8.49088479e-02,   7.20091455e+01, 5.08865823e-02])

I chose to use the middle two features, ctm2 and age, based on the results from using the
SelectKBest, RFE, Chi2 feature selection tools in sklearn. These were consistently ranked
as the most important features from each of the feature selection tools. 

**Classification Results:**

| Classifier | Sensitivity | Specificity | Accuracy | F1 Score | AUC |
|------|------|------|------|------|------|
| Gaussian Naive Bayes | 0.941 | 0.802 | 0.870 | 0.875 | 0.928 |
| Logistic Regression | 0.875 | 0.796 | 0.835 | 0.838 | 0.869 |
| *SVC | 0.984 | 0.776 | 0.877 | 0.886 | 0.933 |

*Best SVC Parameter from Grid Search: Kernel: rbf, C: 10, gamma: 1

By looking at the results from all three classification methods used in part 1, it 
appears that SVC is the best classification method for this data set considering it has
the highest accuracy and AUC. It does, however, sacrifice specificity for an increased
sensitivity. If you want a high accuracy that does not sacrifice specificity as much, 
naive bayes works well on this data set. Logistic regression has the worst performance
across the board.

##Random Forest##
**Classification Results:**

| Classifier | Sensitivity | Specificity | Accuracy | F1 Score | AUC |
|------|------|------|------|------|------|
| Random Forest | 0.938 | 0.912 | 0.924 | 0.923 | 0.979 |

**Feature Importance:**

Random Forest Output: [ 0.19790539  0.2121318   0.54407048  0.04589234]

1. Which features are "most" important? That is, based on your interpretation of the random 
forest output, would you choose to select only a subset of features if you were to use another 
model such as a support vector machine?

This output shows that age is the highest ranking feature, followed by both ctm data points.
The length of stay is the lowest ranking feature and does not provide much insight into
the classification. If I were to use another model based on the random forest output,
I would use the data for the three highest ranking features, age, ctm1 and ctm2.

