"""
LinearRegresion.py

Özgün Yargı

Linear Regression class created by using Numpy library only. Predictor parameters are determined by using Gradient Descent apporach.
Used Objective function is $h(\theta)=\frac{1}{2n}\sum_{i=1}^{n}(\hat{y_i}-y_i)^2, n\epsilon\mathbb{N}$
"""
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Union

np.seterr(all="ignore")

@dataclass(frozen=False, order=True)
class LinearRegression:
    
    iteration_num: int = field(default=100)
    learning_rate: float = field(default=0.01)
    lasso: float = field(default=0)
    ridge: float = field(default=0)
    __weights: np.ndarray = field(default=np.empty(1, dtype="float"), repr=False)
    __X: np.ndarray = field(default=np.empty(1, dtype="float"), repr=False)
    __y: np.ndarray = field(default=np.empty(1, dtype="float"), repr=False)

    def __gradientDescent(self) -> None:
        """
        Optimizes weights
        Used Formula:
        $\theta_j^{(i+1)} = \theta_j^{i}-\alpha\frac{d(h(\theta))}{\theta_j}$
        """
        self.__err = 0
        for sample, truth in zip(self.__X, self.__y): # sum( (y_pred(i)-y_truth(i))*x(i) )
            self.__err += self.__squaredError(sample, truth)*sample # (y_pred(i)-y_truth(i))*x(i)
        self.__weights -= self.learning_rate/self.__X.shape[0]*self.__err + self.learning_rate*self.__ridge_() + self.learning_rate*self.__lasso_() # Theta_i(j+1) = Theta_i(j) - lr/n*sum( (y_pred(i)-y_truth(i))*x(i) )
        self.__weights[-1] -= self.learning_rate*self.__ridge_()[-1] + self.learning_rate*self.__lasso_()[-1] # Do Not Update Theta_0!# Do Not Update Theta_0!

    def __lasso_(self) -> np.ndarray:
        if np.isnan(self.__weights/np.abs(self.__weights)).sum() == 0: 
            return (self.lasso/self.__X.shape[0])*(self.__weights/np.abs(self.__weights))
        else:
            return (self.lasso/self.__X.shape[0])*np.nan_to_num(self.__weights/np.abs(self.__weights), nan=1)
    
    def __ridge_(self) -> np.ndarray:
        return (self.ridge/self.__X.shape[0])*self.__weights


    def __squaredError(self, features:np.ndarray, y_truth:float) -> float:

        return np.dot(features,self.__weights)-y_truth # (y_pred-y_truth)

    def fit(self, X: np.ndarray, y:np.ndarray) -> None:
        self.__X = np.concatenate((np.copy(X), np.ones((X.shape[0],1))), axis=1)
        self.__y = np.copy(y)
        self.__weights = np.array([0.0 for i in range(self.__X.shape[1])])

        for iter in range(self.iteration_num):
            self.__gradientDescent()

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_stack = np.concatenate((np.copy(X), np.ones((X.shape[0],1))), axis=1)
        return X_stack.dot(self.__weights)

def CreateDataset(sampleSize: int=100, dimSize: int=1) -> np.ndarray:
    x = []
    y = []
    for i in range(sampleSize):
        for j in range(2):
            regressor = []
            sample_y = np.random.normal(i*3/2, 1.5)
            for k in range(dimSize):
                sample_x = np.random.normal(i, 3)
                regressor.append(sample_x)
            x.append(regressor)
            y.append([sample_y])
    return np.concatenate((x,y),axis=1, dtype="float")

def rootMeanSquaredError(y_truth: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]) -> float:
    rmse_ = 0
    for indx, pred in enumerate(y_pred):
        rmse_ += (pred-y_truth[indx])**2
    return np.sqrt(rmse_/y_truth.shape[0])

def main():
    rowNum = 2
    colNum = 5
    dataset = CreateDataset(sampleSize=25, dimSize=1)
    fig, ax = plt.subplots(rowNum,colNum,figsize=(colNum*4,rowNum*4),)
    X = dataset[:,:-1]
    y = dataset[:,-1]
    iter = 0
    for i in range(rowNum):
        for j in range(colNum):
            model = LinearRegression(iteration_num=iter, learning_rate=0.001)
            model_ridge = LinearRegression(iteration_num=iter, learning_rate=0.001, ridge=300)
            model_lasso = LinearRegression(iteration_num=iter, learning_rate=0.001, lasso=300) 

            model.fit(X,y)
            model_ridge.fit(X,y)
            model_lasso.fit(X,y)

            preds = model.predict(X)
            model_ridge_preds = model_ridge.predict(X)
            model_lasso_preds = model_lasso.predict(X)

            ax[i][j].scatter(X,y)
            ax[i][j].plot(X, preds, color="r", label=f"W/O Regularization RMSE:{round(rootMeanSquaredError(y,preds), 2)}")
            ax[i][j].plot(X, model_ridge_preds, color="g", label=f"L2 Regularization RMSE:{round(rootMeanSquaredError(y,model_ridge_preds), 2)}")
            ax[i][j].plot(X, model_lasso_preds, color="y", label=f"L1 Regularization RMSE:{round(rootMeanSquaredError(y,model_lasso_preds), 2)}")

            ax[i][j].set_title(f"Iteration Num: {iter}")
            ax[i][j].legend(loc="upper left", fontsize=6)
            iter += 1
    return ax

if __name__ == "__main__":
    np.random.seed(42)
    main()
    plt.savefig("RegressionVisualization.png") 