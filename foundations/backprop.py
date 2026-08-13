import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z = w@x + b
        if activation == "sigmoid":
            return round(1/(1+np.exp(-z)),5)
        else:
            return round(max(0.,z),5)

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
    
        y_pred = self.forward(x,w,b,"sigmoid")
        L = (y_pred-y_true)**2 / 2
        dL_dw = (y_pred-y_true) * y_pred*(1-y_pred)*x
        dL_db = (y_pred-y_true) * y_pred*(1-y_pred)

        return np.round(dL_dw,5),np.round(dL_db,5)
