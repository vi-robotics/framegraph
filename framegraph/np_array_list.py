import numpy as np

class NpArrayList():
    
    def __init__(self, data: np.ndarray, extend_axis=0):
        
        self._extend_axis = extend_axis
        self.data = data
        self._virtual_axis_len = data.shape[extend_axis]

    @property
    def extend_axis(self):
        return self._extend_axis

    @property
    def shape(self):
        res = list(self._backed_shape)
        res[self.extend_axis] = self._virtual_axis_len
        return tuple(res)

    @property
    def size(self):
        return np.prod(self.shape)
    
    @property
    def data(self):
        idx = [np.s_[:] for _ in  self.shape]
        idx[self.extend_axis] = np.s_[:self._virtual_axis_len]
        return self._data[tuple(idx)]

    @data.setter
    def data(self, value):
        if len(value.shape) < self.extend_axis + 1:
            raise ValueError(f"extend_axis {self.extend_axis} is"
                f" incompatible with shape {value.shape}")
        if not isinstance(value, np.ndarray):
            raise ValueError(f"value must be a ndarray, not {type(value)}")
        self._data = value

    def _get_current_idx(self):
        idx = [np.s_[:] for _ in  self.shape]
        idx[self.extend_axis] = self._virtual_axis_len
        return tuple(idx)

    def _grow_array(self):
        new_shape = list(self._backed_shape)
        new_shape[self.extend_axis] *= 2
        self._data.resize(new_shape)

    @property
    def _backed_shape(self):
        return self._data.shape

    @property
    def _backed_size(self):
        return self._data.size

    def append(self, value):
        if self._virtual_axis_len == self._backed_shape[self.extend_axis]:
            self._grow_array()

        self._data[self._get_current_idx()] = value
        self._virtual_axis_len += 1


    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, val):
        res_obj = self.data.__getitem__(val)
        return self.__class__(res_obj, extend_axis=self.extend_axis)
