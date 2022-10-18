import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

@dataclass(frozen=False, order=True)
class LinearRegression:
    
    iteration_num: int = field(default=100)
    learning_rate: float = field(default=0.01)
    lasso: float = field(default=0)
    ridge: float = field(default=0)
    __weights: np.ndarray = field(default=np.empty(1, dtype="float"), repr=False)
    __X: np.ndarray = field(default=np.empty(1, dtype="float"), repr=False)
    __y: np.ndarray = field(default=np.empty(1, dtype="float"), repr=False)

    def __gradientDescent(self):
        self.__err = 0
        for sample, truth in zip(self.__X, self.__y): # sum( (y_pred(i)-y_truth(i))*x(i) )
            self.__err += self.__squaredError(sample, truth)*sample # (y_pred(i)-y_truth(i))*x(i)
        self.__weights -= self.learning_rate/self.__X.shape[0]*self.__err # Theta_i(j+1) = Theta_i(j) - lr/n*sum( (y_pred(i)-y_truth(i))*x(i) )

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

def main():
    model = LinearRegression(iteration_num=20, learning_rate=0.001)
    dataset = CreateDataset(sampleSize=25, dimSize=1)
    X = dataset[:,:-1]
    y = dataset[:,-1]
    model.fit(X,y)
    preds = model.predict(X)
    fig, ax = plt.subplots(1,1,figsize=(8,8))
    ax.scatter(X, y)
    ax.plot(X, preds, color="r")
    return ax

if __name__ == "__main__":
    np.random.seed(42)
    main()
    plt.savefig("RegressionVisualization.png") 