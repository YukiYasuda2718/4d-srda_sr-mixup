from abc import ABCMeta, abstractmethod


class AbstractFftCalculator(metaclass=ABCMeta):
    @abstractmethod
    def apply_fft2(self, grid_data):
        pass

    @abstractmethod
    def apply_ifft2(self, spec_data):
        pass

    @abstractmethod
    def calculate_uv_from_omega(self, grid_omega):
        pass

    @abstractmethod
    def calculate_advection_from_spec_omega(self, spec_omega):
        pass
