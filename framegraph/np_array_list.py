from typing import Tuple, List, Union
from numbers import Number
import numpy as np


class NpArrayList():
    """A wrapper around an ndarray which provides an axis along the array
    a constant time amortized append operation. The axis cannot be changed after
    initialization.
    """

    def __init__(self, data: np.ndarray, extend_axis: int = 0):
        """Initialize a NpArrayList with existing data.

        Args:
            data (np.ndarray): An ndarray to assign to 'data'.
            extend_axis (int, optional): The axis of the ndarray along which
                constant time append will be provided. Defaults to 0.
        """
        self._extend_axis = extend_axis
        self.data = data
        self._virtual_axis_len = data.shape[extend_axis]

    @property
    def extend_axis(self) -> int:
        """The axis of the ndarray which supports constant time append.

        Returns:
            int: The extendable axis.
        """
        return self._extend_axis

    @extend_axis.setter
    def extend_axis(self, axis: int):
        """Set the extend axis of the array.

        Args:
            axis (int): The extend axis of the array.
        """
        if axis >= self._data.ndim:
            raise ValueError(f"axis {axis} is not valid for data of shape"
                             f" {self.shape}")

        self._extend_axis = axis
        self._reset_data()

    @property
    def shape(self) -> Tuple[int, ...]:
        """The virtual shape of the data. This is "virtual" because the real
        array behind the data is likely larger along extend_axis than the
        virtual data.

        Returns:
            Tuple[int, ...]: The shape tuple of the array.
        """
        res: List[int] = list(self._backed_shape)
        res[self.extend_axis] = self._virtual_axis_len
        return tuple(res)

    @property
    def size(self) -> int:
        """The virtual size of the array.

        Returns:
            int: The size of the array.
        """
        return np.prod(self.shape)

    @property
    def data(self) -> np.ndarray:
        """The virtual data of the array. 

        Returns:
            np.ndarray: The data.
        """
        idx = [np.s_[:] for _ in self.shape]
        idx[self.extend_axis] = np.s_[:self._virtual_axis_len]
        return self._data[tuple(idx)]

    @data.setter
    def data(self, value: np.ndarray):
        """Set the virtual data of the array.

        Args:
            value (np.ndarray): The ndarray to set. The extend axis must be less
                than the number of dimensions of the array.

        Raises:
            ValueError: Incompatible extend axis for array shape.
            ValueError: Value is not an ndarray.
        """
        if not isinstance(value, np.ndarray):
            raise ValueError(f"value must be a ndarray, not {type(value)}")
        if value.ndim <= self.extend_axis:
            raise ValueError(f"extend_axis {self.extend_axis} is"
                             f" incompatible with shape {value.shape}")
        self._virtual_axis_len = value.shape[self.extend_axis]
        self._data: np.ndarray = value

    def append(self, value: Union[Number, np.ndarray]):
        """Append the value to the end of the array (along the extend_axis).

        Args:
            value (Union[Number, np.ndarray]): The value must have the same
                shape along every axis as data except extend_axis.
        """
        if self._virtual_axis_len == self._backed_shape[self.extend_axis]:
            self._grow_array()

        self._data[self._get_current_idx()] = value
        self._virtual_axis_len += 1

    def _reset_data(self):
        """Reset the data such that the virtual data is the same as the backed
        data.
        """
        # This might look bizzare, but remember the data property returns the
        # virtual data, and the setter sets the underlying data and handles
        # adjusting the virtual length also!
        self.data = self.data

    def _get_current_idx(self) -> Tuple[Union[int, slice], ...]:
        """Get the index tuple for the current 'append' slice of the array.

        Returns:
            Tuple[Union[int, slice], ...]: The resulting indexing tuple.
        """
        idx: List[Union[int, slice]] = [np.s_[:] for _ in self.shape]
        idx[self.extend_axis] = self._virtual_axis_len
        return tuple(idx)

    def _grow_array(self):
        """Grow the underlying array of the array list. This simply doubles the
        length of the array along extend_axis.
        """
        new_shape = list(self._backed_shape)
        new_shape[self.extend_axis] *= 2
        self._data.resize(new_shape)

    @property
    def _backed_shape(self):
        return self._data.shape

    @property
    def _backed_size(self):
        return self._data.size

    def __str__(self):
        return self.data.__str__()

    def __getitem__(self, val):
        res_obj = self.data.__getitem__(val)
        return self.__class__(res_obj, extend_axis=self.extend_axis)
