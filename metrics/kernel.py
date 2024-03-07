import numpy as np

class Kernel(object):
    def __init__(self):
        pass

    def __call__(self, x, y):
        raise NotImplementedError("Kernel function not implemented")

    def __str__(self):
        return "Default Kernel"

    def __repr__(self):
        return self.__str__()
      
class LinearKernel(Kernel):
    def __init__(self):
        super().__init__()

    def __call__(self, x, y):
        return x @ y.T

    def __str__(self):
        return "Linear Kernel"

    def __repr__(self):
        return self.__str__()
      
      
class PowerKernel(Kernel):
    def __init__(self, degree: int = 2):
        super().__init__()
        self.degree = degree

    def __call__(self, x, y):
        return (x @ y.T + 1) ** self.degree

    def __str__(self):
        return f"Polynomial Kernel of degree {self.degree}"

    def __repr__(self):
        return self.__str__()
      

class RBFKernel(Kernel):
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma

    def __call__(self, x, y):
        return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * self.sigma ** 2))

    def __str__(self):
        return f"RBF Kernel with sigma = {self.sigma}"

    def __repr__(self):
        return self.__str__()