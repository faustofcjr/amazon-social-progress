import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    mean_squared_error, 
    r2_score,
    max_error,
    explained_variance_score   
)


def evaluate_model(model, X, y, X_train, y_train, X_test, y_test, y_pred, cv=10, n_splits=10):
    print("\nModel-evaluation")
    print("----------------------------------------------------------------------")
    
    score = model.score(X_train, y_train)
    print("Score: {:.4f}".format(score))
    
    # cross validataion 
    cv_score = cross_val_score(model, X_train, y_train, cv=cv)
    print("CV train mean score:{:.4f}".format(cv_score.mean()))
    
    # k-fold cross validataion 
    kfold = KFold(n_splits=n_splits, shuffle=True)
    kf_cv_scores = cross_val_score(model, X_train, y_train, cv=kfold)
    print("K-fold CV average score: %.2f" % kf_cv_scores.mean())
    
    # prediction
    r_square = r2_score(y_test, y_pred)
    print("R²: {:.4f}".format(r_square))
    
    mxerr = max_error(y_test, y_pred)
    print("Max Error: {:.4f}".format(mxerr))
    
    ev_score = explained_variance_score(y_test, y_pred)
    print("Explained Variance: {:.4f}".format(ev_score))
    
    mse = mean_squared_error(y_test, y_pred)
    print("MSE: {:.4f}".format(mse))
    
    rmse = mse * (1/2.0)
    print("RMSE: {:.4f}".format(rmse))
    
   
    # x_ax = range(len(y_test))
    # plt.plot(x_ax, y_test, label="Original")
    # plt.plot(x_ax, y_pred, label="Predicted")
    # plt.title("Model test and predicted data")
    # plt.legend()
    # plt.show()


def plot_prediction_result(y_test, y_pred):
    x_ax = range(len(y_test))
    plt.scatter(x_ax, y_test, s=5, color="blue", label="original")
    plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
    plt.legend()
    plt.show()

    
def plot_training_deviance(model, n_estimators, X_test, y_test, y_pred):    
    test_score = np.zeros((n_estimators,), dtype = np.float64)

    for i, y_pred in enumerate(model.staged_predict(X_test)):
        test_score[i] = model.loss_(y_test, y_pred)

    fig = plt.figure(figsize=(6, 6))
    plt.subplot(1, 1, 1)
    
    plt.title("Train and Test Deviance")

    plt.plot(
        np.arange(n_estimators) + 1,
        model.train_score_,
        "b-",
        label="Training set deviance",
    )
    
    plt.plot(
        np.arange(n_estimators) + 1, 
        test_score, 
        "r-", 
        label="Test set deviance"
    )

    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


def plot_feature_importance(model, dataset, X_test, y_test):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)    
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    
    fig = plt.figure(figsize=(30, 10))
    
    plt.subplot(1, 2, 1)
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(dataset.columns)[sorted_idx])
    plt.title("Feature Importance (MDI)")

    result = permutation_importance(
        model, 
        X_test, 
        y_test, 
        n_repeats=10, 
        n_jobs=2
    )
    
    sorted_idx = result.importances_mean.argsort()
    
    plt.subplot(1, 2, 2)
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=np.array(dataset.columns)[sorted_idx],
    )

    plt.title("Permutation Importance (Test set)")
    fig.tight_layout()
    plt.show()