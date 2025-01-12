mod ops;

use num_traits::{One, Zero};

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct Value<T: nalgebra::Scalar, const N: usize> {
    value: T,
    grad: nalgebra::SVector<T, N>,
    #[cfg(feature = "hessian")]
    hess: nalgebra::SMatrix<T, N, N>,
}

impl<T: nalgebra::Scalar, const N: usize> Value<T, N> {
    pub fn value(&self) -> &T {
        &self.value
    }

    pub fn grad(&self) -> &nalgebra::SVector<T, N> {
        &self.grad
    }

    #[cfg(feature = "hessian")]
    pub fn hess(&self) -> &nalgebra::SMatrix<T, N, N> {
        &self.hess
    }

    pub fn new(value: T) -> Self
    where
        T: Zero,
    {
        Self {
            value: value,
            grad: nalgebra::SVector::zeros(),
            #[cfg(feature = "hessian")]
            hess: nalgebra::SMatrix::zeros(),
        }
    }

    pub fn active(value: T, index: usize) -> Self
    where
        T: Zero + One,
    {
        let mut scalar = Self::new(value);
        scalar.grad[index] = T::one();
        scalar
    }
}
