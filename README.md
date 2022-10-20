# Linear Regression with Gradient Descent
![RegressionVisualization](https://user-images.githubusercontent.com/54710793/196934449-e703576b-8f4e-4873-9c3f-4e972ffaa8e5.png)

Linear Regression with Gradient Descent from scratch by using Numpy only.
- Used objective function
$$h(\theta)=\frac{1}{2n} \sum_{i=1}^{n}(\hat{y_i}-y_i)^2, n\epsilon\mathbb{N}$$

$$where, \hat{y}^{(i)} = \theta_0+\theta_1x_1^{(i)}+\theta_2x_2^{(i)}+...+\theta_mx_m^{(i)}$$
- Update Status
$$\theta_j^{(i+1)} = \theta_j^{i}-\alpha\frac{d(h(\theta))}{\theta_j}$$
