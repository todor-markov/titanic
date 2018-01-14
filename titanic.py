import pandas as pd
import numpy as np
from sklearn import linear_model, discriminant_analysis, svm
from sklearn import ensemble, model_selection
from sklearn.preprocessing import scale
from scipy import stats


def get_data_noage(train_df, test_df):
    # The Pclass var is a categorical var, even though it's stored in an int
    train_df.Pclass = train_df.Pclass.astype('object')
    test_df.Pclass = test_df.Pclass.astype('object')

    # Approach 1: remove age data since it is missing for 20% of people

    train_noage = train_df.drop(['Age'], axis=1)
    test_noage = test_df.drop(['Age'], axis=1)

    y_train_noage = train_noage.Survived.as_matrix()
    X_train_noage = (pd
                     .get_dummies(train_noage)
                     .drop(['Survived'], axis=1)
                     .as_matrix())

    X_test_noage = pd.get_dummies(test_noage).as_matrix()
    return (X_train_noage, y_train_noage, X_test_noage)


def run_param_search(model_name, X, y, scale_data=True):
    if scale_data:
        X = scale(X)

    if model_name == 'LogisticRegression':
        model = linear_model.LogisticRegression()
        params = {
            'penalty': ['l1', 'l2'],
            'C': stats.lognorm(s=3),
        }

    if model_name == 'LDA':
        model = discriminant_analysis.LinearDiscriminantAnalysis(solver='lsqr')
        params = {
            'shrinkage': stats.uniform(loc=0, scale=1)
        }

    if model_name == 'QDA':
        model = discriminant_analysis.QuadraticDiscriminantAnalysis()
        params = {
            'reg_param': stats.uniform(loc=0, scale=1)
        }

    if model_name == 'SVM':
        # the polynomial kernel appears to be numerically unstable, and I
        # could not comsistently get if to work
        model = svm.SVC()
        params = {
            'C': stats.lognorm(s=2),
            'kernel': ['rbf', 'sigmoid'],
        }

    if model_name == 'AdaBoost':
        model = ensemble.AdaBoostClassifier()
        params = {
            'n_estimators': stats.randint(low=100, high=1500),
            'learning_rate': stats.uniform(loc=0.5, scale=0.5),
        }

    if model_name == 'GradientBoosting':
        model = ensemble.GradientBoostingClassifier()
        params = {
            'n_estimators': stats.randint(low=100, high=1500),
            'learning_rate': stats.uniform(loc=0.05, scale=0.95),
            'max_depth': stats.randint(low=3, high=8),
            'subsample': stats.uniform(loc=0.5, scale=0.5),
        }

    if model_name == 'RandomForest':
        model = ensemble.RandomForestClassifier()
        params = {
            'n_estimators': stats.randint(low=100, high=1000),
            'max_features': stats.randint(low=1, high=12),
            'min_samples_leaf': stats.randint(low=1, high=10),
        }

    if model_name == 'ExtraTrees':
        model = ensemble.ExtraTreesClassifier()
        params = {
            'n_estimators': stats.randint(low=100, high=1000),
            'max_features': stats.randint(low=1, high=12),
            'min_samples_leaf': stats.randint(low=1, high=10),
        }

    param_search = (model_selection
                    .RandomizedSearchCV(estimator=model,
                                        param_distributions=params,
                                        n_iter=200,
                                        cv=10,
                                        n_jobs=4,
                                        return_train_score=True,
                                        verbose=1))

    param_search.fit(X, y)
    best_param_indices = np.argsort(-param_search
                                    .cv_results_['mean_test_score'])[0:10]

    return (param_search, best_param_indices)


def get_correlation(estimators, X, y):
    predictions = []
    for est in estimators:
        est.fit(X, y)
        predictions.append(est.predict(X))

    predictions_matrix = np.column_stack(predictions)
    corr_matrix = np.corrcoef(predictions_matrix, rowvar=False)

    return corr_matrix


def repeated_cross_validate(estimator, X, y, cv=10, num_repeats=20):
    total_scores = []
    for i in range(num_repeats):
        scores = model_selection.cross_val_score(estimator=model,
                                                 X=X_train,
                                                 y=y_train,
                                                 cv=10,
                                                 verbose=1)

        total_scores.append(np.mean(scores))

    return total_scores


if __name__ == '__main__':

    # PassengerId, ticket number and names are irrelevant for survival
    # Cabin data is missing for most people (77%)

    train = (pd.read_csv('data/train.csv')
             .drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1))

    test = (pd.read_csv('data/test.csv')
            .drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1))

    # All test samples have an 'Embarked' field; only 2 train samples lack one
    # Thus, we can safely exclude those 2 samples

    train = train.dropna(subset=['Embarked'])

    X_train, y_train, X_test = get_data_noage(train, test)

    X_train = scale(X_train)

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

    SVM = svm.SVC(C=1.546)

    AdaBoost = ensemble.AdaBoostClassifier(n_estimators=1298)

    GradientBoosting = (ensemble
                        .GradientBoostingClassifier(learning_rate=0.094,
                                                    max_depth=6,
                                                    n_estimators=866,
                                                    subsample=0.95))

    RandomForest = (ensemble
                    .RandomForestClassifier(max_features=4,
                                            min_samples_leaf=3,
                                            n_estimators=424))

    estimators = [('svm', SVM),
                  ('ada', AdaBoost),
                  ('gb', GradientBoosting),
                  ('rf', RandomForest)]

    model = ensemble.VotingClassifier(estimators=estimators,
                                      voting='hard',
                                      n_jobs=4)

    model.fit(X_train, y_train)

    # There is 1 test sample without a Fare variable (sample #152). Since
    # all our algorithms use the Fare variable, we'll just ignore this sample
    # and mark it as not surviving

    X_test = np.delete(X_test, (152), axis=0)

    X_test = scale(X_test)
    y_test_predict = model.predict(X_test)
    y_test_predict = np.insert(y_test_predict, (152), 0, axis=0)

    predictions = np.column_stack([np.array(range(892, 1310)), y_test_predict])
    np.savetxt('data/test_predictions.csv', predictions,
               fmt='%d', delimiter=',',
               header='PassengerId,Survived', comments='')
