import jax, jax.numpy as jnp
from functools import partial

from jaxtyping import jaxtyped, Array, Float 
from beartype import beartype

@jax.jit(static_argnames=('i', 'd'))
@jaxtyped(typechecker=beartype)
def cxdb(x: Float[Array, ''], knots: Float[Array, 'k'], i: int, d: int) -> Float[Array, '']:
    '''Cox-De-Boor (JIT Compatible)
    Args:
        x: input `[]`
        knots: knots `[k]`
        i: basis index `int`
        d: degree `int`
    Returns:
        `[]` evaluation of the basis function in `x` 
    '''

    if d == 0:
        is_last_interval = (i == len(knots) - 2)
        in_interval = (knots[i] <= x) & (x < knots[i+1])
        at_right_end = (x == knots[-1])
        return jnp.where(in_interval | (is_last_interval & at_right_end), 1.0, 0.0)

    a, b = knots[i+d] - knots[i], knots[i+d+1] - knots[i+1]

    left = jnp.where(a != 0, (x - knots[i]) / jnp.where(a != 0, a, 1.0) * cxdb(x, knots, i, d-1), 0.0)
    right = jnp.where(b != 0, (knots[i+d+1] - x) / jnp.where(b != 0, b, 1.0) * cxdb(x, knots, i+1, d-1), 0.0)

    return left + right

@jax.jit(static_argnames=('i', 'd'))
@jaxtyped(typechecker=beartype)
def dcxdb(x: Float[Array, ''], knots: Float[Array, 'k'], i: int, d: int) -> Float[Array, '']:
    '''Derivative of Cox-De-Boor
    Args:
        x: input `[]`
        knots: knots `[k]`
        i: basis index `int`
        d: degree `int`
    Returns:
        `[]` evaluation of the derivative basis function in `x` 
    '''
    a, b = knots[i+d] - knots[i], knots[i+d+1] - knots[i+1]
    left = jnp.where(a != 0, d / jnp.where(a != 0, a, 1.0) * cxdb(x, knots, i, d-1), 0.0)
    right = jnp.where(b != 0, d / jnp.where(b != 0, b, 1.0) * cxdb(x, knots, i+1, d-1), 0.0)
    return left - right

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def repeat_knots(knots: Float[Array, 'k'], d: int) -> Float[Array, 'K']:
    '''Repeat boundary knots d times on each end.
    Args:
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[k+d+d]` knots with repeated start and end 
    '''
    left = jnp.repeat(knots[0], d)
    right = jnp.repeat(knots[-1], d)
    return jnp.concat([left, knots, right])

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_matrix_row(x: Float[Array, ''], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    '''Evaluate the basis functions for a single input.
    Args:
        x: point `[]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[n]` evaluation of `x` in all basis functions
    '''
    x = jnp.where(x >= knots[-1], knots[-1] - 1e-6, x)
    n_basis = len(knots) - d - 1
    return jnp.stack([cxdb(x, knots, i, d) for i in range(n_basis)])

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_dmatrix_row(x: Float[Array, ''], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    '''Evaluate the basis function derivatives for a single input.
    Args:
        x: point `[]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[n]` evaluation of `x` in all derivative basis functions
    '''
    x = jnp.where(x >= knots[-1], knots[-1] - 1e-6, x)
    x = jnp.where(x >= knots[-1], knots[-1] - 1e-6, x)
    n_basis = len(knots) - d - 1
    return jnp.stack([dcxdb(x, knots, i, d) for i in range(n_basis)])

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_matrix(x: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'N n']:
    '''Evaluate the basis functions for N inputs.
    Args:
        x: points `[N]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[N n]` design matrix for inputs `x`
    '''
    return jax.vmap(partial(design_matrix_row, knots=knots, d=d))(x)

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_dmatrix(x: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'N n']:
    '''Evaluate the basis function derivatives for N inputs.
    Args:
        x: points `[N]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[N n]` derivative design matrix for inputs `x`
    '''
    return jax.vmap(partial(design_dmatrix_row, knots=knots, d=d))(x)

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def fit_bspline_coefs(x: Float[Array, 'N'], y: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    '''Solve the least-squares problem for the B-splines coefficients.
    Args:
        x: points `[N]`
        y: targets `[N]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[n]` fitted coefficients
    '''
    B = design_matrix(x, knots, d)
    return jnp.linalg.lstsq(B, y)[0]

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def fit_bspline_dcoefs(x: Float[Array, 'N'], y: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    '''Solve the least-squares problem for the B-splines derivatives coefficients.
    Args:
        x: points `[N]`
        y: targets `[N]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[n]` fitted derivative coefficients
    '''
    B = design_dmatrix(x, knots, d)
    return jnp.linalg.lstsq(B, y)[0]

@jax.jit(static_argnames='d')
def bspline_inference(
    x: Float[Array, 'N'], 
    c: Float[Array, 'n'],
    knots: Float[Array, 'k'],
    d: int,
) -> Float[Array, 'N']:
    '''Compute the B-spline inference for input `x`.
    Args:
        x: input `[N]`
        c: coefficients `[n]`
        knots: knots `[k]`
        d: degree `int`
    Returns:
        `[N]` spline(xi) for xi in x
    '''
    return jnp.squeeze(design_matrix(jnp.atleast_1d(x), knots, d) @ c)
