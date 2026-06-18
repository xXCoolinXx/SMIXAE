# GroupJumpReLU derivation

## Preliminaries

Let $H(x)$ be the Heaviside step function, i.e. 
$$H(x) = \begin{cases}1 & x > 0 \\ 0 & x \leq 0\end{cases}$$

The derivative of $H(x)$ is defined as $\delta(x)$, the Dirac delta function. Since the Dirac delta function contains a singularity at $0$, we instead use the approximate function $\Pi_\epsilon(x)$, where $\epsilon \in \mathbb{R}^+$ is called the **bandwidth**. 

Specifically, 
$$\Pi_\epsilon(x) = \frac{R_\epsilon(x)}{\epsilon}$$
where 
$$R_\epsilon(x) = \begin{cases} 1 & -\frac{\epsilon}{2} < x < \frac{\epsilon}{2} \\ 0 & \text{otherwise} \end{cases}$$

## Forward Pass

Let $\theta \in \mathbb{R}$, $x \in \mathbb{R}^n$, and let $\sigma(x,\theta) : \mathbb{R}^n \times \mathbb{R} \to \mathbb{R}^n$ be the activation function. 

We define the activation function as follows

$$\sigma(x,\theta) = x H(||x|| - \theta)$$

where $||x||$ is the $L_2$ norm. 

## Straight Through Estimator

We cannot work with the derivative of this function, since the derivative is the Dirac delta function which doesn't play nice with SGD. Instead, we use the approximate function $\Pi_\epsilon(x)$ defined above. 

First, we need to define the derivative with respect to $x$. We simply assume that we may ignore any dependence on the inner term of the step function, and define the derivative as 
$$\nabla_x \sigma(x,\theta) = H(||x|| -\theta)$$

Next, we must derive the derivative with respect to $\theta$. We see
$$\partial_\theta \sigma(x,\theta) = - x \delta(||x|| - \theta) = - \hat{x} ||x|| \delta(||x|| - \theta) = - \hat{x} \theta \delta(||x|| - \theta) \approx - \hat{x} \theta \Pi_\epsilon(||x|| - \theta) $$

Where we make use of the sifting property of the Dirac delta function. Notably, the biggest diference in comparison with standard JumpReLU is the introduction of the $\hat{x}$ term, which projects the direction of the gradient based on the direction of $x$ itself. 

## Ensuring Positive $\theta$

To ensure that $\theta$ is positive, let $\theta = e^t$, where $t \in \mathbb{R}$. $t$ becomes the trainable parameter. Autograd will figure this out, it will be okay. 