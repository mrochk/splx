import jax, jax.numpy as jnp
from splx.bspline import *

def f(x): return jnp.cos(4*jnp.pi*x)

def main():
    for i in range(10):

        x = jnp.linspace(0, 1)
        y = f(x)

        knots = repeat_knots3(jnp.linspace(0, 1, 40))
        print(knots)

        B = design_matrix3(x, knots)

        c = fit_bspline_coefs3(x, y, knots)

        inf = B @ c
        print(y)
        print(inf)

if __name__ == "__main__":
    main()
