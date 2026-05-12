import jax, jax.numpy as jnp
from functools import partial

from jaxtyping import jaxtyped, Array, Float 
from beartype import beartype

'''
d -> B-spline degree (order - 1)
k -> number of knots (for the full knots vector)
n = k-d-1 -> number of control points (coefficients)
N -> number of data points
'''

@jax.jit(static_argnames=('i', 'd'))
@jaxtyped(typechecker=beartype)
def cxdb(x: Float[Array, ''], knots: Float[Array, 'k'], i: int, d: int) -> Float[Array, '']:
    '''Cox-De-Boor (JIT Compatible)'''

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
    """Derivative of Cox-De-Boor"""
    a, b = knots[i+d] - knots[i], knots[i+d+1] - knots[i+1]
    left = jnp.where(a != 0, d / jnp.where(a != 0, a, 1.0) * cxdb(x, knots, i, d-1), 0.0)
    right = jnp.where(b != 0, d / jnp.where(b != 0, b, 1.0) * cxdb(x, knots, i+1, d-1), 0.0)
    return left - right

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def repeat_knots(knots: Float[Array, 'k'], d: int) -> Float[Array, 'K']:
    """Repeat boundary knots d times on each end."""
    left = jnp.repeat(knots[0], d)
    right = jnp.repeat(knots[-1], d)
    return jnp.concat([left, knots, right])

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_matrix_row(x: Float[Array, ''], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    '''Evaluate the basis functions for a single input.'''
    x = jnp.where(x >= knots[-1], knots[-1] - 1e-6, x)
    n_basis = len(knots) - d - 1
    return jnp.stack([cxdb(x, knots, i, d) for i in range(n_basis)])

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_dmatrix_row(x: Float[Array, ''], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    '''Evaluate the basis function derivatives for a single input.'''
    x = jnp.where(x >= knots[-1], knots[-1] - 1e-6, x)
    x = jnp.where(x >= knots[-1], knots[-1] - 1e-6, x)
    n_basis = len(knots) - d - 1
    return jnp.stack([dcxdb(x, knots, i, d) for i in range(n_basis)])

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_matrix(x: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'N n']:
    '''Evaluate the basis functions for N inputs.'''
    return jax.vmap(partial(design_matrix_row, knots=knots, d=d))(x)

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def design_dmatrix(x: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'N n']:
    '''Evaluate the basis function derivatives for N inputs.'''
    return jax.vmap(partial(design_dmatrix_row, knots=knots, d=d))(x)

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def fit_bspline_coefs(x: Float[Array, 'N'], y: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    B = design_matrix(x, knots, d)
    return jnp.linalg.lstsq(B, y)[0]

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def fit_bspline_dcoefs(x: Float[Array, 'N'], y: Float[Array, 'N'], knots: Float[Array, 'k'], d: int) -> Float[Array, 'n']:
    B = design_dmatrix(x, knots, d)
    return jnp.linalg.lstsq(B, y)[0]

@jax.jit(static_argnames='d')
@jaxtyped(typechecker=beartype)
def bspline_inference(x: Float[Array, ''], c: Float[Array, 'n'], knots: Float[Array, 'k'], d: int) -> Float[Array, '']:
    return design_matrix_row(x, knots, d) @ c
