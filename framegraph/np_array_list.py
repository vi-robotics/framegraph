import numpy as np

class NpArrayList(np.ndarray):
    
    def __new__(subtype, shape, dtype=float, buffer=None, offset=0,
                strides=None, order=None, extend_axis=0):
        obj = super().__new__(subtype, shape, dtype,
                              buffer, offset, strides, order)
        obj._extend_axis = extend_axis
        obj._virtual_axis_len = obj._backed_shape[extend_axis]
        return obj

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
    
    def _grow_array(self):
        new_shape = list(self._backed_shape)
        new_shape[self.extend_axis] *= 2
        self.resize(new_shape, refcheck=False)

    @property
    def _backed_shape(self):
        return super().shape

    @property
    def _backed_size(self):
        return super().size()

    def append(self, value):
        if self._virtual_axis_len == self._backed_shape[self.extend_axis]:
            self._grow_array()
        
        shape = self.shape
        idx = [np.s_[:] for _ in shape]
        idx[self.extend_axis] = self._virtual_axis_len
        
        self[tuple(idx)] = value
        self._virtual_axis_len += 1


    def __getitem__(self, val):
        res_obj = super().__getitem__(val)
        try:
            res_obj._virtual_axis_len = res_obj._backed_shape[self.extend_axis]
        except AttributeError:
            pass
        return res_obj


    def __array_finalize__(self, obj):
        if obj is None: return
        self._extend_axis = getattr(obj, 'extend_axis', 0)
        self._virtual_axis_len = getattr(obj, '_virtual_axis_len', 0)