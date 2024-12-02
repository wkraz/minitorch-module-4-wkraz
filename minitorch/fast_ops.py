from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Any

import numpy as np
from numba import prange
from numba import njit as _njit

from .tensor_data import (
    MAX_DIMS,
    broadcast_index,
    index_to_position,
    shape_broadcast,
    to_index,
)
from .tensor_ops import MapProto, TensorOps

if TYPE_CHECKING:
    from typing import Callable, Optional

    from .tensor import Tensor
    from .tensor_data import Index, Shape, Storage, Strides

# TIP: Use `NUMBA_DISABLE_JIT=1 pytest tests/ -m task3_1` to run these tests without JIT.

# This code will JIT compile fast versions your tensor_data functions.
# If you get an error, read the docs for NUMBA as to what is allowed
# in these functions.
Fn = TypeVar("Fn")


def njit(fn: Fn, **kwargs: Any) -> Fn:
    return _njit(inline="always", **kwargs)(fn)  # type: ignore


to_index = njit(to_index)
index_to_position = njit(index_to_position)
broadcast_index = njit(broadcast_index)


class FastOps(TensorOps):
    @staticmethod
    def map(fn: Callable[[float], float]) -> MapProto:
        """See `tensor_ops.py`"""
        # This line JIT compiles your tensor_map
        f = tensor_map(njit(fn))

        def ret(a: Tensor, out: Optional[Tensor] = None) -> Tensor:
            if out is None:
                out = a.zeros(a.shape)
            f(*out.tuple(), *a.tuple())
            return out

        return ret

    @staticmethod
    def zip(fn: Callable[[float, float], float]) -> Callable[[Tensor, Tensor], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_zip(njit(fn))

        def ret(a: Tensor, b: Tensor) -> Tensor:
            c_shape = shape_broadcast(a.shape, b.shape)
            out = a.zeros(c_shape)
            f(*out.tuple(), *a.tuple(), *b.tuple())
            return out

        return ret

    @staticmethod
    def reduce(
        fn: Callable[[float, float], float], start: float = 0.0
    ) -> Callable[[Tensor, int], Tensor]:
        """See `tensor_ops.py`"""
        f = tensor_reduce(njit(fn))

        def ret(a: Tensor, dim: int) -> Tensor:
            out_shape = list(a.shape)
            out_shape[dim] = 1

            # Other values when not sum.
            out = a.zeros(tuple(out_shape))
            out._tensor._storage[:] = start

            f(*out.tuple(), *a.tuple(), dim)
            return out

        return ret

    @staticmethod
    def matrix_multiply(a: Tensor, b: Tensor) -> Tensor:
        """Batched tensor matrix multiply ::

            for n:
              for i:
                for j:
                  for k:
                    out[n, i, j] += a[n, i, k] * b[n, k, j]

        Where n indicates an optional broadcasted batched dimension.

        Should work for tensor shapes of 3 dims ::

            assert a.shape[-1] == b.shape[-2]

        Args:
        ----
            a : tensor data a
            b : tensor data b

        Returns:
        -------
            New tensor data

        """
        # Make these always be a 3 dimensional multiply
        both_2d = 0
        if len(a.shape) == 2:
            a = a.contiguous().view(1, a.shape[0], a.shape[1])
            both_2d += 1
        if len(b.shape) == 2:
            b = b.contiguous().view(1, b.shape[0], b.shape[1])
            both_2d += 1
        both_2d = both_2d == 2

        ls = list(shape_broadcast(a.shape[:-2], b.shape[:-2]))
        ls.append(a.shape[-2])
        ls.append(b.shape[-1])
        assert a.shape[-1] == b.shape[-2]
        out = a.zeros(tuple(ls))

        tensor_matrix_multiply(*out.tuple(), *a.tuple(), *b.tuple())

        # Undo 3d if we added it.
        if both_2d:
            out = out.view(out.shape[1], out.shape[2])
        return out


# Implementations


def tensor_map(
    fn: Callable[[float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides], None]:
    """NUMBA low_level tensor_map function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out` and `in` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function mappings floats-to-floats to apply.

    Returns:
    -------
        Tensor map function.

    """

    def _map(
        out: Storage,
        out_shape: Shape,
        out_strides: Strides,
        in_storage: Storage,
        in_shape: Shape,
        in_strides: Strides,
    ) -> None:
        out_size = len(out)
        
        # optimization: if shapes and strides are aligned, we don't need to compute indices
        if np.allclose(out_shape, in_shape) and np.allclose(out_strides, in_strides):
            # directly apply the function with parallel execution
            for i in prange(out_size):
                out[i] = fn(in_storage[i])                                          # just apply function directly
        else:
            # not aligned so we have to actually calculate indices
            out_index = np.zeros_like(out_shape)
            in_index = np.zeros_like(in_shape)
            
            for i in prange(out_size):
                to_index(i, out_shape, out_index)                                   # convert flat index to multidim
                broadcast_index(out_index, out_shape, in_shape, in_index)           # broadcast
                out[i] = fn(in_storage[index_to_position(in_index, in_strides)])    # apply function 

    return njit(_map, parallel=True)  # type: ignore


def tensor_zip(
    fn: Callable[[float, float], float],
) -> Callable[
    [Storage, Shape, Strides, Storage, Shape, Strides, Storage, Shape, Strides], None
]:
    """NUMBA higher-order tensor zip function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * When `out`, `a`, `b` are stride-aligned, avoid indexing

    Args:
    ----
        fn: function maps two floats to float to apply.

    Returns:
    -------
        Tensor zip function.

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
        out_size = len(out)
        
        # same optimization as map, don't compute indices if aligned
        if (np.allclose(out_shape, a_shape) and np.allclose(out_strides, a_strides)) and \
            (np.allclose(a_shape, b_shape) and np.allclose(a_strides, b_strides)):
                # directly apply fn to a and b with parallel execution
                for i in prange(out_size):
                    out[i] = fn(a_storage[i], b_storage[i])
        else:
            # not aligned so we have to compute indices :(
            a_index = np.zeros_like(a_shape)
            b_index = np.zeros_like(b_shape)
            out_index = np.zeros_like(out_shape)
            
            # parallel computing
            for i in prange(out_size):
                # get multidim index
                to_index(i, out_shape, out_index)
                # broadcast to a and b
                broadcast_index(out_index, out_shape, a_shape, a_index)
                broadcast_index(out_index, out_shape, b_shape, b_index)
                # apply fn to a and b
                out[i] = fn(
                    a_storage[index_to_position(a_index, a_strides)],
                    b_storage[index_to_position(b_index, b_strides)]
                )
                

    return njit(_zip, parallel=True)  # type: ignore


def tensor_reduce(
    fn: Callable[[float, float], float],
) -> Callable[[Storage, Shape, Strides, Storage, Shape, Strides, int], None]:
    """NUMBA higher-order tensor reduce function. See `tensor_ops.py` for description.

    Optimizations:

    * Main loop in parallel
    * All indices use numpy buffers
    * Inner-loop should not call any functions or write non-local variables

    Args:
    ----
        fn: reduction function mapping two floats to float.

    Returns:
    -------
        Tensor reduce function

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
        out_size = len(out)
        out_index = np.zeros_like(out_shape)
        
        # we don't have to check if the tensors are aligned here
        for i in prange(out_size):
            # get multi dim index
            to_index(i, out_shape, out_index)
            # base position in input tensor
            a_position = index_to_position(out_index, a_strides)
            # initialize with first value
            out_val = a_storage[a_position]
            
            # reduce along specified dimension
            for _ in range(1, a_shape[reduce_dim]):
                # move along reduction dimension
                a_position += a_strides[reduce_dim]
                # combine values
                out_val = fn(out_val, a_storage[a_position])
            # store the result we just calculated
            out[i] = out_val 
                    

    return njit(_reduce, parallel=True)  # type: ignore


def _tensor_matrix_multiply(
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
    """NUMBA tensor matrix multiply function.

    Should work for any tensor shapes that broadcast as long as

    ```
    assert a_shape[-1] == b_shape[-2]
    ```

    Optimizations:

    * Outer loop in parallel
    * No index buffers or function calls
    * Inner loop should have no global writes, 1 multiply.


    Args:
    ----
        out (Storage): storage for `out` tensor
        out_shape (Shape): shape for `out` tensor
        out_strides (Strides): strides for `out` tensor
        a_storage (Storage): storage for `a` tensor
        a_shape (Shape): shape for `a` tensor
        a_strides (Strides): strides for `a` tensor
        b_storage (Storage): storage for `b` tensor
        b_shape (Shape): shape for `b` tensor
        b_strides (Strides): strides for `b` tensor

    Returns:
    -------
        None : Fills in `out`

    """
    # remember, matrix multiplication def:
    #   sum(A[i, k] * B[k, j])
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0

    # parallel execution over all elements in out (c)
    for out_pos in prange(len(out)):
        # compute row and column with modulo arithmetic
        out_row = (out_pos // out_strides[-2] % out_shape[-2])
        out_col = (out_pos // out_strides[-1] % out_shape[-1])
        # get batch index for higher dim tensors
        out_batch = out_pos // out_strides[0] if len(out_shape) > 2 else 0
        
        # compute a & b base positions
        a_pos = out_batch * a_batch_stride + out_row * a_strides[-2]        # starting memory pos in A for curr row in A
        b_pos = out_batch * b_batch_stride + out_col * b_strides[-1]        # starting memory pos in B for curr col in B
        
        # inner loop summation
        curr_sum = 0.0
        for _ in range(b_shape[-2]):
            # compute dot product and add to sum
            curr_sum += a_storage[a_pos] * b_storage[b_pos]
            # change a&b_pos accordingly
            a_pos += a_strides[-1]
            b_pos += b_strides[-2]
        # update result in storage
        out[out_pos] = curr_sum


tensor_matrix_multiply = njit(_tensor_matrix_multiply, parallel=True)
assert tensor_matrix_multiply is not None
