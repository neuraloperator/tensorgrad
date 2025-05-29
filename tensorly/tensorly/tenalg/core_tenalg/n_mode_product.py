from ... import backend as T
from ... import unfold, fold, vec_to_tensor
import torch

def mode_dot(tensor, matrix_or_vector, mode, transpose=False, handle_complex_half_by_upcasting=False):
    """n-mode product of a tensor and a matrix or vector at the specified mode

    Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`

    Parameters
    ----------
    tensor : ndarray
        tensor of shape ``(i_1, ..., i_k, ..., i_N)``
    matrix_or_vector : ndarray
        1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
        matrix or vectors to which to n-mode multiply the tensor
    mode : int
    transpose : bool, default is False
        If True, the matrix is transposed.
        For complex tensors, the conjugate transpose is used.
    handle_complex_half_by_upcasting : bool, default is False
        If True, handle complex half tensors by upcasting to complex64.
    
    Returns
    -------
    ndarray
        `mode`-mode product of `tensor` by `matrix_or_vector`
        * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
        * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector

    See also
    --------
    multi_mode_dot : chaining several mode_dot in one call
    """
    # the mode along which to fold might decrease if we take product with a vector
    fold_mode = mode
    new_shape = list(tensor.shape)

    if T.ndim(matrix_or_vector) == 2:  # Tensor times matrix
         # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned in mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[dim]} (dim 1 of matrix)"
            )
            
        if transpose and matrix_or_vector.dtype == torch.complex32:
            matrix_or_vector = T.conj(T.transpose(matrix_or_vector))
            
        new_shape[mode] = matrix_or_vector.shape[0]
        vec = False

    elif T.ndim(matrix_or_vector) == 1:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned for mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[0]} (vector size)"
            )
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            new_shape = ()
        vec = True

    else:
        raise ValueError(
            "Can only take n_mode_product with a vector or a matrix."
            f"Provided array of dimension {T.ndim(matrix_or_vector)} not in [1, 2]."
        )

    unfolded = unfold(tensor, mode)
    if tensor.dtype == torch.complex32 and not handle_complex_half_by_upcasting:
        # first try with torch.matmul
        try:
            res = torch.matmul(matrix_or_vector, unfolded)
        except RuntimeError:
            # Split complex32 into real and imaginary half tensors
            tensor_real = unfolded.real
            tensor_imag = unfolded.imag
            mat_or_vec = matrix_or_vector
            mat_real = mat_or_vec.real
            mat_imag = mat_or_vec.imag
            # Compute real and imaginary parts using matmul/dot on half-precision floats
            if vec:
                result_real = T.dot(mat_real, tensor_real) - T.dot(mat_imag, tensor_imag)
                result_imag = T.dot(mat_real, tensor_imag) + T.dot(mat_imag, tensor_real)
            else:
                result_real = T.matmul(mat_real, tensor_real) - T.matmul(mat_imag, tensor_imag)
                result_imag = T.matmul(mat_real, tensor_imag) + T.matmul(mat_imag, tensor_real)
            # Combine back to complex32 using PyTorch-specific call
            res = torch.complex(result_real, result_imag).to(torch.complex32)
    else:
        if handle_complex_half_by_upcasting and tensor.dtype == torch.complex32:
            matrix_or_vector = matrix_or_vector.to(torch.complex64)
        # Use backend's dot for other dtypes
        res = T.dot(matrix_or_vector, unfolded)
        

    if vec: # We contracted with a vector, leading to a vector
        return vec_to_tensor(res, shape=new_shape)
    else:  # tensor times vec: refold the unfolding
        return fold(res, fold_mode, new_shape)


def multi_mode_dot(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False):
    """n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    matrix_or_vec_list : list of matrices or vectors of length ``tensor.ndim``

    skip : None or int, optional, default is None
        If not None, index of a matrix to skip.
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        If True, the matrices or vectors in in the list are transposed.
        For complex tensors, the conjugate transpose is used.

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list at mode `mode`

    Notes
    -----
    If no modes are specified, just assumes there is one matrix or vector per mode and returns:

    :math:`\\text{tensor  }\\times_0 \\text{ matrix or vec list[0] }\\times_1 \\cdots \\times_n \\text{ matrix or vec list[n] }`

    See also
    --------
    mode_dot
    """
    if modes is None:
        modes = range(len(matrix_or_vec_list))

    decrement = 0  # If we multiply by a vector, we diminish the dimension of the tensor

    res = tensor

    # Order of mode dots doesn't matter for different modes
    # Sorting by mode shouldn't change order for equal modes
    factors_modes = sorted(zip(matrix_or_vec_list, modes), key=lambda x: x[1])
    for i, (matrix_or_vec, mode) in enumerate(factors_modes):
        if (skip is not None) and (i == skip):
            continue

        if transpose:
            res = mode_dot(res, T.conj(T.transpose(matrix_or_vec)), mode - decrement)
        else:
            res = mode_dot(res, matrix_or_vec, mode - decrement)

        if T.ndim(matrix_or_vec) == 1:
            decrement += 1

    return res


def multi_mode_dot_einsum(tensor, matrix_or_vec_list, modes=None, skip=None, transpose=False, handle_complex_half_by_upcasting=False):
    """n-mode product of a tensor and several matrices or vectors using a single einsum operation

    This is an optimized version of multi_mode_dot that uses a single einsum call instead of
    sequential mode_dot operations, which can be significantly faster and more memory-efficient
    for certain tensor shapes and hardware.

    Parameters
    ----------
    tensor : ndarray
        Input tensor to contract

    matrix_or_vec_list : list of matrices or vectors of length ``tensor.ndim``
        The matrices or vectors to contract with the tensor

    modes : None or int list, optional, default is None
        Modes to contract with each matrix or vector
        If None, uses range(len(matrix_or_vec_list))

    skip : None or int, optional, default is None
        If not None, index of a matrix to skip.

    transpose : bool, optional, default is False
        If True, the matrices or vectors are transposed.
        For complex tensors, the conjugate transpose is used.

    handle_complex_half_by_upcasting : bool, default is False
        If True, handle complex half tensors by upcasting to complex64.

    Returns
    -------
    ndarray
        tensor times each matrix or vector in the list at specified modes

    Notes
    -----
    This implementation is generally faster than sequential mode_dot operations,
    especially for larger tensors, but requires PyTorch's einsum implementation.
    """
    if modes is None:
        modes = list(range(len(matrix_or_vec_list)))
    
    # Handle skip parameter
    if skip is not None:
        matrix_or_vec_list = [m for i, m in enumerate(matrix_or_vec_list) if i != skip]
        modes = [m for i, m in enumerate(modes) if i != skip]
    
    # Handle complex32 tensors if needed
    is_complex32 = tensor.dtype == torch.complex32
    if handle_complex_half_by_upcasting and is_complex32:
        tensor = tensor.to(torch.complex64)
        matrix_or_vec_list = [m.to(torch.complex64) if m.is_complex() else m for m in matrix_or_vec_list]
    
    # Preprocess matrices if transpose=True
    if transpose:
        matrix_or_vec_list = [
            T.conj(T.transpose(m)) if T.ndim(m) == 2 and getattr(m, 'is_complex', lambda: False)()
            else T.transpose(m) if T.ndim(m) == 2
            else m
            for m in matrix_or_vec_list
        ]
    
    # 1) Build subscript for the input tensor: e.g. 'abcd'
    rank = tensor.ndim
    in_subs = [chr(ord('a') + i) for i in range(rank)]
    
    # 2) For each factor, create its subscripts and adjust output subscripts
    op_subs = []
    out_subs = in_subs.copy()
    
    # Track how many dimensions were removed to adjust mode indices
    removed_dims = 0
    adjusted_modes = sorted(modes)  # Sort to process from lowest to highest mode
    
    for i, (factor, mode) in enumerate(zip(matrix_or_vec_list, adjusted_modes)):
        # Adjust mode based on dimensions already removed by vector contractions
        adj_mode = mode - removed_dims
        
        letter = in_subs[mode]  # Original letter for this mode
        
        if T.ndim(factor) == 2:
            # For matrices: use uppercase letter for the new dimension
            new_letter = letter.upper()
            op_subs.append(f"{new_letter}{letter}")
            out_subs[adj_mode] = new_letter
        else:
            # For vectors: use the original letter (but it will be contracted away)
            op_subs.append(letter)
            out_subs.pop(adj_mode)
            removed_dims += 1
    
    # Form the einsum equation
    equation = "".join(in_subs) + "," + ",".join(op_subs) + "->" + "".join(out_subs)
    
    # Perform the einsum contraction
    try:
        result = torch.einsum(equation, tensor, *matrix_or_vec_list)
    except RuntimeError as e:
        if is_complex32 and not handle_complex_half_by_upcasting:
            # If complex32 failed, try again with upcasting
            tensor_64 = tensor.to(torch.complex64)
            factors_64 = [m.to(torch.complex64) if m.is_complex() else m for m in matrix_or_vec_list]
            result = torch.einsum(equation, tensor_64, *factors_64).to(torch.complex32)
        else:
            raise e
    
    # Convert back to complex32 if needed
    if handle_complex_half_by_upcasting and is_complex32:
        result = result.to(torch.complex32)
    
    return result
