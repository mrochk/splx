import jax, jax.numpy as jnp
from bsplx import *

def f(x): return jnp.cos(4*jnp.pi*x)

def main():
    for i in range(10):

        x = jnp.linspace(0, 1)
        y = f(x)

        knots = repeat_knots(jnp.linspace(0, 1, 40), 3)
        print(knots)

        B = design_matrix(x, knots, 3)

        c = fit_bspline_coefs(x, y, knots, 3)

        inf = B @ c
        print(y)
        print(inf)

if __name__ == "__main__":
    main()
