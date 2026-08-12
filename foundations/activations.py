import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        arr = np.empty_like(z)
        arr[:] = 1/(1+np.exp(-z))
        return np.round(arr,5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        lis = []
        for i in z:
            ans = max(0.0,i)
            lis.append(ans)
        return lis
