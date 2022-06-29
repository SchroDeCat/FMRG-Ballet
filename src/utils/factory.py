from typing import List
import numpy as np
from matplotlib import pyplot as plt


class Data_Factory:
    """
    Collections of different objective functions
    """

    def __generate_config(self, dim, num):
        """Generate required num & dim config following uniform distribution"""
        if dim == 1:
            self.config = np.random.uniform(low=[-1], high=[1], size=[num, dim])
            self.config = np.hstack([self.config, np.zeros([num, dim])])
        elif dim > 1:
            self.config = np.random.uniform(
                low=-1 * np.ones(dim), high=1 * np.ones(dim), size=[num, dim]
            )

    def convex_1(self, dim: int = 3, num: int = 1000) -> np:
        """
        simple d-dim convex function
        """
        self.__generate_config(dim=dim, num=num)
        self.target_value = np.sum(self.config ** 2, axis=1)[:, np.newaxis]
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    def convex_2(self, dim: int = 3, num: int = 1000) -> np:
        """
        simple d-dim convex function for optimization
        """
        self.__generate_config(dim=dim, num=num)
        self.target_value = -np.sum(self.config ** 2, axis=1)[:, np.newaxis]
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    def example_fn(self, dim: int = 3, num: int = 1000) -> np.ndarray:
        """
        f = lambda x: np.sin(10*x[..., 0]) * np.exp(-x[..., 0]**2)
        """
        self.__generate_config(dim=dim, num=num)
        self.target_value = np.sum(
            np.sin(10 * self.config) * (np.exp(-self.config) ** 2), axis=1
        )
        self.data = np.hstack([self.config, self.target_value[:, np.newaxis]])
        return self.data

    def exmaple_fn_1d(self, dim: int = 1, num: int = 1000) -> np.ndarray:
        """DKL 1d example"""
        assert dim == 1
        self.__generate_config(dim=dim, num=num)
        x = self.config
        self.target_value = (x + 0.5 >= 0) * np.sin(64 * (x + 0.5) ** 4)
        self.data = np.hstack(
            [self.config, self.target_value[:, 0].reshape([num, dim])]
        )
        return self.data

    def robot_arm_8d(self, num: int = 10000) -> np.ndarray:
        """https://www.sfu.ca/~ssurjano/robot.html"""
        assert num >= 1
        dim = 8
        self.__generate_config(dim=dim, num=num)
        self.config = (self.config + 1) / 2
        self.config[:, :4] = self.config[:, :4] * np.pi
        u = np.zeros([num, 1])
        v = np.zeros([num, 1])
        for i in range(4):
            tmp = np.sum(self.config[:, : i + 1], axis=1).reshape([num, 1])
            u += self.config[:, i + 4].reshape([num, 1]) * np.cos(tmp)
            v += self.config[:, i + 4].reshape([num, 1]) * np.sin(tmp)
        self.target_value = np.sqrt(u ** 2 + v ** 2)
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    def rastrigin(self, dim=4, num: int = 1000) -> np.ndarray:
        """https://www.sfu.ca/~ssurjano/rastr.html"""
        assert dim > 1
        self.__generate_config(dim=dim, num=num)
        self.config = self.config * 5.12
        self.tmp = self.config ** 2 - 10 * np.cos(2 * np.pi * self.config)
        self.target_value = 10 * dim + np.sum(self.tmp, axis=1).reshape([num, 1])
        # raise(Exception("Not implemented!"))
        self.data = np.hstack([self.config, self.target_value])
        return self.data

    @staticmethod
    def nearest(np_data, point: List) -> np:
        """
        Tool to find nearest data point
        """
        length = np.size(point)
        diff = np.abs(np_data[:, :length] - np.array(point))
        index = np.argmin(np.sum(diff, axis=1))
        return np_data[index], index

    def obj_func(self, test_p: list) -> float:
        """For query druing optimization"""
        input_dim = np.shape(test_p)[0]
        if input_dim != np.size(test_p) and input_dim > 1:
            value = [
                Data_Factory.nearest(self.data, test_p[i, :])[-1]
                for i in range(input_dim)
            ]
            return value
        else:
            data_point = Data_Factory.nearest(self.data, test_p)
            return data_point[-1]

    def plot_1Dsample(self):
        plt.scatter(self.data[:, 0], self.data[:, -1])
        plt.title("1-D Demo of Original Data")
        plt.xlabel("Config")
        plt.ylabel("Target Value")
