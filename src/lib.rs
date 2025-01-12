mod ops;

use nalgebra::ArrayStorage;
use num_traits::{ConstZero, One, Zero};

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

    pub const fn new_const(value: T) -> Self
    where
        T: ConstZero + Copy,
    {
        Self {
            value: value,
            grad: nalgebra::SVector::from_array_storage(ArrayStorage([[T::ZERO; N]])),
            #[cfg(feature = "hessian")]
            hess: nalgebra::SMatrix::from_array_storage(ArrayStorage([[T::ZERO; N]; N])),
        }
    }

    pub fn active(mut self, index: usize) -> Self
    where
        T: Zero + One,
    {
        self.grad[index] = T::one();
        self
    }
}
