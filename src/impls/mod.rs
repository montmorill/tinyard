mod arithmetic;
mod compare;
mod complex;
mod real;

use std::ops::{AddAssign, MulAssign};

use num_traits::{One, Zero};

use crate::Tin;

impl<T: nalgebra::Scalar, const N: usize> Tin<T, N> {
    fn chain(
        &self,
        value: T,                            // f
        grad: T,                             // df/dx
        #[cfg(feature = "hessian")] hess: T, // ddf/dxx
    ) -> Self
    where
        T: AddAssign + MulAssign + Zero + One,
    {
        Self {
            value,
            #[cfg(feature = "hessian")]
            hess: &self.grad * self.grad.transpose() * hess + &self.hess * grad.clone(),
            grad: &self.grad * grad,
        }
    }
}

impl<T: nalgebra::Scalar, const N: usize> std::fmt::Display for Tin<T, N>
where
    T: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tin[{}]", self.value)
    }
}
