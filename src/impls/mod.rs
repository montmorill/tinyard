mod arithmetic;

use std::ops::{AddAssign, MulAssign};

use num_traits::{One, Zero};

use crate::Value;

impl<T: nalgebra::Scalar, const N: usize> Value<T, N> {
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
