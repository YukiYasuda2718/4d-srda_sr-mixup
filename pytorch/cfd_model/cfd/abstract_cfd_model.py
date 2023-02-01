from abc import ABCMeta, abstractmethod


class AbstractCfdModel(metaclass=ABCMeta):
    @property
    @abstractmethod
    def time(self):
        pass

    @property
    @abstractmethod
    def vorticity(self):
        pass

    @property
    @abstractmethod
    def state_size(self):
        pass

    @abstractmethod
    def initialize(self, t0, omega0):
        pass

    @abstractmethod
    def calc_grid_data(self):
        pass

    @abstractmethod
    def time_integrate(self, dt: float, nt: int):
        pass