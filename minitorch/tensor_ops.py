from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional, Type

import numpy as np
from typing_extensions import Protocol

from . import operators
from .cuda_ops import _tensor_matrix_multiply
from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)

if TYPE_CHECKING:
    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides
    

class MapProto(Protocol):
    def __call__(self, x: Tensor, out: Optional[Tensor] = ..., /) -> Tensor:
        ...


class TensorOps:
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        pass

    @staticmethod
    def cmap(fn: Callable[[float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        pass

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        pass

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        # check dimensions
        assert a.shape[-1] == b.shape[-2], "Shapes incompatible for matmul"
        
        # Create output tensor shape
        out_shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
        out = a.zeros(out_shape)

        # Perform matrix multiplication (optimized implementation)
        a_data, b_data, out_data = a._tensor, b._tensor, out._tensor
        _tensor_matrix_multiply(
            out_data._storage, out_data._shape, out_data._strides,
            out_data.size, a_data._storage, a_data._shape, a_data._strides,
            b_data._storage, b_data._shape, b_data._strides
        )
        return out

    cuda = False


class TensorBackend:
    def __init__(self, ops: Type[TensorOps]):
        """
        Dynamically construct a tensor backend based on a `tensor_ops` object
        that implements map, zip, and reduce higher-order functions.

        Args:
            ops : tensor operations object see `tensor_ops.py`


        Returns :
            A collection of tensor functions

        """

        # Maps
        self.neg_map = ops.map(operators.neg)
        self.sigmoid_map = ops.map(operators.sigmoid)
        self.relu_map = ops.map(operators.relu)
        self.log_map = ops.map(operators.log)
        self.exp_map = ops.map(operators.exp)
        self.id_map = ops.map(operators.id)
        self.id_cmap = ops.cmap(operators.id)
        self.inv_map = ops.map(operators.inv)

        # Zips
        self.add_zip = ops.zip(operators.add)
        self.mul_zip = ops.zip(operators.mul)
        self.lt_zip = ops.zip(operators.lt)
        self.eq_zip = ops.zip(operators.eq)
        self.is_close_zip = ops.zip(operators.is_close)
        self.relu_back_zip = ops.zip(operators.relu_back)
        self.log_back_zip = ops.zip(operators.log_back)
        self.inv_back_zip = ops.zip(operators.inv_back)

        # Reduce
        self.add_reduce = ops.reduce(operators.add, 0.0)
        self.mul_reduce = ops.reduce(operators.mul, 1.0)
        self.matrix_multiply = ops.matrix_multiply
        self.cuda = ops.cuda


class SimpleOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """
        Higher-order tensor map function ::

          fn_map = map(fn)
          fn_map(a, out)
          out

        Simple version::

            for i:
                for j:
                    out[i, j] = fn(a[i, j])

        Broadcasted version (`a` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0])

        Args:
            fn: function from float-to-float to apply.
            a (:class:`TensorData`): tensor to map over
            out (:class:`TensorData`): optional, tensor data to fill in,
                   should broadcast with `a`

        Returns:
            new tensor data
        """

        f = tensor_map(fn)

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(
        fn: Callable[[float, float], float]
    ) -> Callable[["Tensor", "Tensor"], "Tensor"]:
        """
        Higher-order tensor zip function ::

          fn_zip = zip(fn)
          out = fn_zip(a, b)

        Simple version ::

            for i:
                for j:
                    out[i, j] = fn(a[i, j], b[i, j])

        Broadcasted version (`a` and `b` might be smaller than `out`) ::

            for i:
                for j:
                    out[i, j] = fn(a[i, 0], b[0, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to zip over
            b (:class:`TensorData`): tensor to zip over

        Returns:
            :class:`TensorData` : new tensor data
        """

        f = tensor_zip(fn)

        def ret(a: "Tensor", b: "Tensor") -> "Tensor":
            if a.shape != b.shape:
                c_shape = shape_broadcast(a.shape, b.shape)
            else:
                c_shape = a.shape
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[["Tensor", int], "Tensor"]:
        """
        Higher-order tensor reduce function. ::

          fn_reduce = reduce(fn)
          out = fn_reduce(a, dim)

        Simple version ::

            for j:
                out[1, j] = start
                for i:
                    out[1, j] = fn(out[1, j], a[i, j])


        Args:
            fn: function from two floats-to-float to apply
            a (:class:`TensorData`): tensor to reduce over
            dim (int): int of dim to reduce

        Returns:
            :class:`TensorData` : new tensor
        """
        f = tensor_reduce(fn)

        def ret(a: "Tensor", dim: int) -> "Tensor":
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: "Tensor", b: "Tensor") -> "Tensor":
        # Check compatibility of shapes
        assert a.shape[-1] == b.shape[-2], "Shapes incompatible for matmul"

        # Create output tensor shape
        out_shape = (*a.shape[:-2], a.shape[-2], b.shape[-1])
        out = a.zeros(out_shape)

        # Extract data
        a_data, b_data, out_data = a._tensor, b._tensor, out._tensor

        # Iterate over all combinations of output indices
        out_index = np.zeros(len(out_shape), dtype=int)
        a_index = np.zeros(len(a.shape), dtype=int)
        b_index = np.zeros(len(b.shape), dtype=int)

        for ordinal in range(out_data.size):
            # Convert ordinal to multidimensional index for output
            to_index(ordinal, out_shape, out_index)

            # Calculate positions for `a` and `b` using broadcast logic
            dot_product = 0.0
            for k in range(a.shape[-1]):
                for i in range(len(out_index)):
                    a_index[i] = out_index[i]
                    b_index[i] = out_index[i]
                a_index[-1] = k
                b_index[-2] = k

                a_pos = index_to_position(a_index, a_data._strides)
                b_pos = index_to_position(b_index, b_data._strides)
                dot_product += a_data._storage[a_pos] * b_data._storage[b_pos]

            # Write the result into the output tensor
            out_pos = index_to_position(out_index, out_data._strides)
            out_data._storage[out_pos] = dot_product

        return out

    is_cuda = False


# Implementations.


def tensor_map(fn: Callable[[float], float]) -> Any:
    """
    Low-level implementation of tensor map between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `in_storage` assuming `out_shape` and `in_shape`
      broadcast. (`in_shape` must be smaller than `out_shape`).

    Args:
        fn: function from float-to-float to apply
        out (array): storage for out tensor
        out_shape (array): shape for out tensor
        out_strides (array): strides for out tensor
        in_storage (array): storage for in tensor
        in_shape (array): shape for in tensor
        in_strides (array): strides for in tensor

    Returns:
        None : Fills in `out`
    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_index = np.empty(len(out_shape), dtype=int)
        in_index = np.empty(len(in_shape), dtype=int)

        for ordinal in range(len(out)):
            to_index(ordinal, out_shape, out_index)

            broadcast_index(out_index, out_shape, in_shape, in_index)

            out_position = index_to_position(out_index, out_strides)
            in_position = index_to_position(in_index, in_strides)

            out[out_position] = fn(in_storage[in_position])

    return _map


def tensor_zip(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor zip between
    tensors with *possibly different strides*.

    Simple version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `out_shape`
      and `a_shape` are the same size.

    Broadcasted version:

    * Fill in the `out` array by applying `fn` to each
      value of `a_storage` and `b_storage` assuming `a_shape`
      and `b_shape` broadcast to `out_shape`.

    Args:
        fn: function mapping two floats to float to apply
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        b_storage (array): storage for `b` tensor
        b_shape (array): shape for `b` tensor
        b_strides (array): strides for `b` tensor

    Returns:
        None : Fills in `out`
    """

    def _zip(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        b_storage: Storage,
        b_shape: Shape,
        b_strides: Strides,
    ) -> None:
        out_index = np.empty(len(out_shape), dtype=int)
        a_index = np.empty(len(a_shape), dtype=int)
        b_index = np.empty(len(b_shape), dtype=int)

        for ordinal in range(len(out)):
            to_index(ordinal, out_shape, out_index)
            
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)

            out_position = index_to_position(out_index, out_strides)
            a_position = index_to_position(a_index, a_strides)
            b_position = index_to_position(b_index, b_strides)

            out[out_position] = fn(a_storage[a_position], b_storage[b_position])

    return _zip


def tensor_reduce(fn: Callable[[float, float], float]) -> Any:
    """
    Low-level implementation of tensor reduce.

    * `out_shape` will be the same as `a_shape`
       except with `reduce_dim` turned to size `1`

    Args:
        fn: reduction function mapping two floats to float
        out (array): storage for `out` tensor
        out_shape (array): shape for `out` tensor
        out_strides (array): strides for `out` tensor
        a_storage (array): storage for `a` tensor
        a_shape (array): shape for `a` tensor
        a_strides (array): strides for `a` tensor
        reduce_dim (int): dimension to reduce out

    Returns:
        None : Fills in `out`
    """

    def _reduce(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        a_storage: Storage,
        a_shape: Shape,
        a_strides: Strides,
        reduce_dim: int,
    ) -> None:
        out_index = np.empty(len(out_shape), dtype=int)
        a_index = np.empty(len(a_shape), dtype=int)

        for ordinal in range(len(out)):
            to_index(ordinal, out_shape, out_index)

            out_position = index_to_position(out_index, out_strides)

            current_value = None

            for i in range(a_shape[reduce_dim]):
                for j in range(len(out_index)):
                    a_index[j] = out_index[j]
                a_index[reduce_dim] = i

                a_position = index_to_position(a_index, a_strides)

                if current_value is None:
                    current_value = a_storage[a_position]
                else:
                    current_value = fn(current_value, a_storage[a_position])

            out[out_position] = current_value

    return _reduce


SimpleBackend = TensorBackend(SimpleOps)