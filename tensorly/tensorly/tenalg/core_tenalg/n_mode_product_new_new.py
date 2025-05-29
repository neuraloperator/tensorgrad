from ... import backend as T
from ... import unfold, fold, vec_to_tensor
import torch

def log_memory(message):
    """Log GPU memory usage at a specific point"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"TENSOR_OP MEMORY [{message}] - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Max: {max_allocated:.2f}MB")

def mode_dot(tensor, matrix_or_vector, mode, transpose=False, handle_complex_half_by_upcasting=False, out=None, skip_shape_check=False):
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
    out : ndarray, optional
        If provided, the result will be stored in this array. 
        If not provided, a new array will be allocated.
    skip_shape_check : bool, default is False
        If True, skip the output buffer shape compatibility check.
        This is useful when called from multi_mode_dot which manages buffers.
    
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
    # Identify the type of multiplication (matrix or vector)
    new_shape = list(tensor.shape)
    is_vector_case = T.ndim(matrix_or_vector) == 1
    is_matrix_case = T.ndim(matrix_or_vector) == 2
    
    if is_matrix_case:  # Tensor times matrix
        # Test for the validity of the operation
        dim = 0 if transpose else 1
        if matrix_or_vector.shape[dim] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned in mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[dim]} (dim {dim} of matrix)"
            )
            
        if transpose and matrix_or_vector.dtype == torch.complex32:
            matrix_or_vector = T.conj(T.transpose(matrix_or_vector))
            
        new_shape[mode] = matrix_or_vector.shape[0]

    elif is_vector_case:  # Tensor times vector
        if matrix_or_vector.shape[0] != tensor.shape[mode]:
            raise ValueError(
                f"shapes {tensor.shape} and {matrix_or_vector.shape} not aligned for mode-{mode} multiplication: "
                f"{tensor.shape[mode]} (mode {mode}) != {matrix_or_vector.shape[0]} (vector size)"
            )
        if len(new_shape) > 1:
            new_shape.pop(mode)
        else:
            new_shape = ()

    else:
        raise ValueError(
            "Can only take n_mode_product with a vector or a matrix."
            f"Provided array of dimension {T.ndim(matrix_or_vector)} not in [1, 2]."
        )

    # Check output buffer shape compatibility early if provided and not skipping check
    if out is not None and not skip_shape_check:
        # Convert new_shape to tuple for comparison
        expected_shape = tuple(new_shape)
        if out.shape != expected_shape:
            raise ValueError(
                f"Output buffer shape mismatch: expected {expected_shape}, got {out.shape}. "
                f"This may happen in multi_mode_dot when mode indices change due to vector contractions."
            )

    # Memory-efficient implementation with permute+reshape instead of unfold/fold
    # Move the target mode to the front, then reshape into a matrix
    t_perm = T.moveaxis(tensor, mode, 0)
    i_k = t_perm.shape[0]
    flat = T.reshape(t_perm, (i_k, -1))  # View operation, no copy

    # Prepare matrix/vector for matmul - handle transpose
    mat = matrix_or_vector
    if transpose:
        if hasattr(mat, 'is_complex') and mat.is_complex():
            mat = T.conj(T.transpose(mat))
        else:
            mat = T.transpose(mat)

    # Handle complex half-precision tensors
    is_complex32 = tensor.dtype == torch.complex32
    
    # Create a buffer for the matmul result if needed
    matmul_out = None
    if out is not None:
        # If we're going to reshape the result later, we need a temporary buffer
        # for the matmul result with correct shape (flat shape)
        if is_vector_case:
            matmul_shape = new_shape  # Vector case: result is already in final shape
        else:
            # Matrix case: result will be reshaped/permuted
            rest_shape = t_perm.shape[1:]
            matmul_shape = (mat.shape[0],) + rest_shape
            
        # For vector case, we can sometimes use the output directly
        if is_vector_case and out.shape == (mat.shape[0] * flat.shape[1],):
            matmul_out = out  # Can use output directly as matmul buffer
    
    if is_complex32 and not handle_complex_half_by_upcasting:
        # Try native complex32 matmul first (modern GPUs support this)
        try:
            if matmul_out is not None:
                # Direct matmul into output buffer
                torch.matmul(mat, flat, out=matmul_out)
                res = matmul_out
            else:
                res = T.matmul(mat, flat)
        except RuntimeError:
            # Fallback to manual real/imag computation
            tensor_real = flat.real
            tensor_imag = flat.imag
            mat_real = mat.real
            mat_imag = mat.imag
            
            # Matrix multiplication with complex numbers
            # For complex case, we need separate real/imag calculations
            if matmul_out is not None:
                # Need separate real and imag outputs
                real_out = torch.empty_like(matmul_out, dtype=torch.float16)
                imag_out = torch.empty_like(matmul_out, dtype=torch.float16)
                
                # Real part: real*real - imag*imag
                torch.matmul(mat_real, tensor_real, out=real_out)
                torch.matmul(mat_imag, tensor_imag, out=imag_out)
                real_out.sub_(imag_out)
                
                # Imag part: real*imag + imag*real
                torch.matmul(mat_real, tensor_imag, out=imag_out)
                torch.addmm(imag_out, mat_imag, tensor_real)
                
                # Combine into complex result
                res = torch.complex(real_out, imag_out).to(torch.complex32)
                del real_out, imag_out
            else:
                result_real = T.matmul(mat_real, tensor_real) - T.matmul(mat_imag, tensor_imag)
                result_imag = T.matmul(mat_real, tensor_imag) + T.matmul(mat_imag, tensor_real)
                res = torch.complex(result_real, result_imag).to(torch.complex32)
            
    elif handle_complex_half_by_upcasting and is_complex32:
        # Upcast to complex64 for more robust kernels
        mat_64 = mat.to(torch.complex64)
        flat_64 = flat.to(torch.complex64)
        
        if matmul_out is not None:
            # Use a temporary buffer in complex64
            temp_out = torch.empty(matmul_out.shape, dtype=torch.complex64, device=matmul_out.device)
            torch.matmul(mat_64, flat_64, out=temp_out)
            # Downcast back to complex32
            res = temp_out.to(torch.complex32, non_blocking=True)
            del temp_out
        else:
            res = torch.matmul(mat_64, flat_64).to(torch.complex32)
            
        del mat_64, flat_64
    else:
        # Standard path: matmul handles both matrix×matrix and vector×matrix
        if matmul_out is not None:
            # Direct matmul into output buffer
            torch.matmul(mat, flat, out=matmul_out)
            res = matmul_out
        else:
            res = T.matmul(mat, flat)

    # Reshape the result back to the tensor format
    if is_vector_case:  # We contracted with a vector, leading to a vector
        if res.shape != new_shape:  # Only reshape if necessary
            result = T.reshape(res, new_shape)
        else:
            result = res
    else:  # tensor times matrix: reshape and permute back
        rest_shape = t_perm.shape[1:]
        if (res.shape[0] != mat.shape[0]) or (res.shape[1:] != rest_shape):
            result_temp = T.reshape(res, (mat.shape[0],) + rest_shape)
        else:
            result_temp = res
        
        if result_temp.shape != (mat.shape[0],) + rest_shape:
            # Safety check to ensure reshape worked correctly
            raise ValueError(f"Reshaped tensor has wrong shape: {result_temp.shape} vs expected {(mat.shape[0],) + rest_shape}")
            
        result = T.moveaxis(result_temp, 0, mode)
        
    # Copy to output buffer if provided and not already using it
    if out is not None and result is not out:
        # Double-check shapes before copy if not skipping shape check
        if not skip_shape_check and result.shape != out.shape:
            raise RuntimeError(
                f"Shape mismatch in final copy: result shape {result.shape}, out shape {out.shape}. "
                f"This indicates an internal error in the mode_dot calculation."
            )
            
        # When skip_shape_check is True, we need to handle the case where shapes don't match
        if skip_shape_check and result.shape != out.shape:
            # Return the result directly instead of trying to copy to out
            return result
            
        # Only copy if shapes match
        out.copy_(result)
        return out
    
    return result


def multi_mode_dot(tensor,
                   factors,
                   modes=None,
                   skip=None,
                   transpose=False,
                   handle_complex_half_by_upcasting=False,
                   out=None):
    """n-mode product of a tensor and several matrices or vectors over several modes

    Parameters
    ----------
    tensor : ndarray

    factors : list of matrices or vectors of length ``tensor.ndim``

    skip : None or int, optional, default is None
        If not None, index of a matrix to skip.
        Note that in any case, `modes`, if provided, should have a length of ``tensor.ndim``

    modes : None or int list, optional, default is None

    transpose : bool, optional, default is False
        If True, the matrices or vectors in in the list are transposed.
        For complex tensors, the conjugate transpose is used.
        
    handle_complex_half_by_upcasting : bool, default is False
        If True, handle complex half tensors by upcasting to complex64.
        
    out : ndarray, optional
        If provided, the result will be stored in this array.
        If not provided, a new array will be allocated.

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
    # 1) normalize modes list
    n = len(factors)
    if modes is None:
        modes = list(range(n))
    else:
        modes = list(modes)
    
    # 2) build ops list, skipping if needed
    ops = []
    for i, (mat, mode) in enumerate(zip(factors, modes)):
        if skip is None or i != skip:
            ops.append([mat, mode])
    
    # If no operations to perform, return tensor as is or copy to output buffer
    if not ops:
        if out is not None:
            out.copy_(tensor)
            return out
        return tensor
    
    # 3) simulate to get the final shape and adjust modes for vector removals
    final_shape = list(tensor.shape)
    for idx, (mat, mode) in enumerate(ops):
        if T.ndim(mat) == 2:
            final_shape[mode] = mat.shape[0]
        else:
            # vector: drop that axis
            final_shape.pop(mode)
            # decrement subsequent modes
            for j in range(idx+1, len(ops)):
                if ops[j][1] > mode:
                    ops[j][1] -= 1
    
    # 4) validate `out` if provided
    if out is not None:
        if tuple(out.shape) != tuple(final_shape):
            # Print a clear warning about the mismatch
            print(f"Warning: Output buffer shape {tuple(out.shape)} doesn't match final shape {tuple(final_shape)}. A new tensor will be returned.")
            # Don't use the provided buffer at all during processing
            out = None
    
    # 5) now do the contractions, re-using a scratch buffer
    res = tensor
    scratch = None
    
    for idx, (mat, mode) in enumerate(ops):
        is_last = (idx == len(ops)-1)
        
        # compute this step's shape
        cur_shape = list(res.shape)
        if T.ndim(mat) == 2:
            cur_shape[mode] = mat.shape[0]
        else:
            cur_shape.pop(mode)
        
        # pick target buffer - only use out on the last step if shapes match
        if is_last and out is not None:
            # Double check that shapes will match
            target_shape = tuple(cur_shape)
            if out.shape == target_shape:
                target = out
            else:
                # Allocate a new buffer for the last step since out doesn't match
                target = torch.empty(cur_shape, dtype=res.dtype, device=res.device)
        else:
            # Allocate/reuse scratch buffer for intermediate steps
            if scratch is None or tuple(scratch.shape) != tuple(cur_shape):
                scratch = torch.empty(cur_shape, dtype=res.dtype, device=res.device)
            target = scratch
        
        # handle transpose
        if transpose:
            if hasattr(mat, 'is_complex') and mat.is_complex():
                m = T.conj(T.transpose(mat))
            else:
                m = T.transpose(mat)
        else:
            m = mat
        
        # perform the mode_dot into target buffer
        res = mode_dot(
            res,
            m,
            mode,
            transpose=False,  # already applied
            handle_complex_half_by_upcasting=handle_complex_half_by_upcasting,
            out=target,
            skip_shape_check=True  # Let multi_mode_dot manage buffer shapes
        )
        
        # adjust remaining modes on vector contraction
        if T.ndim(mat) == 1:
            for j in range(idx+1, len(ops)):
                if ops[j][1] > mode:
                    ops[j][1] -= 1
    
    # If we have a result and an unused output buffer with matching shape, copy now
    if out is not None and res is not out and res.shape == out.shape:
        out.copy_(res)
        return out
        
    return res
