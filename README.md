# titanic
Code for Kaggle's Titanic survival prediction competition

I ran hyperparameter optimization on the following methods:
    
Logistic Regression, Linear Discriminant Analysis (LDA),
Quadratic Discriminant Analysis (QDA), Support Vector Machines (SVM),
AdaBoost, Gradient Boosted Trees, Random Forest, Extra Trees
    
The results are recorded in the file hyperparameter_tuning_results.txt
The file follows the following style:

Algorithm name
Training accuracies for the 10 parameter groups with best val scores
Validation accuracies for the top 10 parameter groups
Parameter settings for the top 10 parameter groups
    
I then evaluated the correlation of the different models and tested
several voting models that combined the first-order models that had
been tuned earlier.
    
That test showed the highest average cross-validation accuracy was
obtained by a voting classifier that combines an SVM, AdaBoost,
Gradient Boosting and a Random Forest
