use crate::Value;
use nalgebra::{ClosedAddAssign, ClosedDivAssign, ClosedMulAssign, ClosedSubAssign};
use num_traits::{One, Zero};
use simba::scalar::ClosedNeg;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

macro_rules! binary_op_impl {
    ($Trait:ident $method:ident $TraitAssign:ident $method_assign:ident $($($bound:tt)+)?) => {
        impl<T: nalgebra::Scalar, const N: usize> $Trait for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($bound)*)?,
        {
            type Output = Value<T, N>;

            #[inline]
            fn $method(mut self, rhs: Self) -> Self::Output {
                self.$method_assign(Borrowed {
                    value: rhs.value,
                    grad: &rhs.grad,
                    #[cfg(feature = "hessian")]
                    hess: &rhs.hess,
                });
                self
            }
        }

        impl<T: nalgebra::Scalar, const N: usize> $Trait<&Value<T, N>> for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($bound)*)?,
        {
            type Output = Value<T, N>;

            #[inline]
            fn $method(mut self, rhs: &Self) -> Self::Output {
                self.$method_assign(Borrowed {
                    value: rhs.value.clone(),
                    grad: &rhs.grad,
                    #[cfg(feature = "hessian")]
                    hess: &rhs.hess,
                });
                self
            }
        }

        impl<T: nalgebra::Scalar, const N: usize> $Trait<Value<T, N>> for &Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($bound)*)?,
        {
            type Output = Value<T, N>;

            #[inline]
            fn $method(self, rhs: Value<T, N>) -> Self::Output {
                rhs.$method(self)
            }
        }

        impl<T: nalgebra::Scalar, const N: usize> $TraitAssign for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($bound)*)?,
        {
            #[inline]
            fn $method_assign(&mut self, rhs: Self) {
                self.$method_assign(Borrowed {
                    value: rhs.value,
                    grad: &rhs.grad,
                    #[cfg(feature = "hessian")]
                    hess: &rhs.hess,
                });
            }
        }

        impl<T: nalgebra::Scalar, const N: usize> $TraitAssign<&Self> for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($bound)*)?,
        {
            #[inline]
            fn $method_assign(&mut self, rhs: &Self) {
                self.$method_assign(Borrowed {
                    value: rhs.value.clone(),
                    grad: &rhs.grad,
                    #[cfg(feature = "hessian")]
                    hess: &rhs.hess,
                });
            }
        }
    };
}

struct Borrowed<'a, T: nalgebra::Scalar, const N: usize> {
    value: T,
    grad: &'a nalgebra::SVector<T, N>,
    #[cfg(feature = "hessian")]
    hess: &'a nalgebra::SMatrix<T, N, N>,
}

impl<T: nalgebra::Scalar, const N: usize> Neg for Value<T, N>
where
    T: ClosedNeg,
{
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            grad: -self.grad,
            #[cfg(feature = "hessian")]
            hess: -self.hess,
        }
    }
}

impl<'a, T: nalgebra::Scalar, const N: usize> AddAssign<Borrowed<'a, T, N>> for Value<T, N>
where
    T: ClosedAddAssign,
{
    fn add_assign(&mut self, rhs: Borrowed<'a, T, N>) {
        self.value += rhs.value;
        self.grad += rhs.grad;
        #[cfg(feature = "hessian")]
        {
            self.hess += rhs.hess;
        }
    }
}

impl<'a, T: nalgebra::Scalar, const N: usize> SubAssign<Borrowed<'a, T, N>> for Value<T, N>
where
    T: ClosedSubAssign,
{
    fn sub_assign(&mut self, rhs: Borrowed<'a, T, N>) {
        self.value -= rhs.value;
        self.grad -= rhs.grad;
        #[cfg(feature = "hessian")]
        {
            self.hess -= rhs.hess;
        }
    }
}

impl<'a, T: nalgebra::Scalar, const N: usize> MulAssign<Borrowed<'a, T, N>> for Value<T, N>
where
    T: ClosedAddAssign + ClosedMulAssign + Zero + One,
{
    fn mul_assign(&mut self, rhs: Borrowed<'a, T, N>) {
        let mut lhs = unsafe { std::mem::MaybeUninit::<Value<T, N>>::uninit().assume_init() };
        std::mem::swap(self, &mut lhs);
        self.value = lhs.value.clone() * rhs.value.clone();
        #[cfg(not(feature = "hessian"))]
        {
            self.grad = lhs.grad * rhs.value + rhs.grad * lhs.value;
        }
        #[cfg(feature = "hessian")]
        {
            self.grad = lhs.grad.clone() * rhs.value.clone() + rhs.grad * lhs.value.clone();
            self.hess = lhs.hess * rhs.value
                + rhs.grad * lhs.grad.transpose()
                + lhs.grad * rhs.grad.transpose()
                + rhs.hess * lhs.value;
        }
    }
}

impl<'a, T: nalgebra::Scalar, const N: usize> DivAssign<Borrowed<'a, T, N>> for Value<T, N>
where
    T: ClosedAddAssign + ClosedSubAssign + ClosedMulAssign + ClosedDivAssign + Zero + One,
{
    fn div_assign(&mut self, rhs: Borrowed<'a, T, N>) {
        let mut lhs = unsafe { std::mem::MaybeUninit::<Value<T, N>>::uninit().assume_init() };
        std::mem::swap(self, &mut lhs);
        self.value = lhs.value.clone() / rhs.value.clone();
        #[cfg(not(feature = "hessian"))]
        {
            self.grad = (lhs.grad * rhs.value.clone() - rhs.grad * lhs.value)
                / (rhs.value.clone() * rhs.value);
        }
        #[cfg(feature = "hessian")]
        {
            self.grad = (lhs.grad * rhs.value.clone() - rhs.grad * lhs.value)
                / (rhs.value.clone() * rhs.value.clone());
            self.hess = (lhs.hess
                - self.grad.clone() * rhs.grad.transpose()
                - rhs.grad * self.grad.transpose()
                - rhs.hess * self.value.clone())
                / rhs.value;

            // self.grad = (lhs.grad * rhs.value.clone() - rhs.grad * lhs.value.clone())
            //     / (rhs.value.clone() * rhs.value.clone());
            // self.hess = (lhs.hess
            //     - self.grad.clone() * rhs.grad.transpose()
            //     - rhs.grad * self.grad.transpose())
            //     / rhs.value
            //     - rhs.hess * lhs.value;
        }
    }
}

binary_op_impl!(Add add AddAssign add_assign);
binary_op_impl!(Sub sub SubAssign sub_assign);
binary_op_impl!(Mul mul MulAssign mul_assign Zero + One + AddAssign);
binary_op_impl!(Div div DivAssign div_assign Zero + One + AddAssign + ClosedSubAssign + MulAssign);

impl<T: nalgebra::Scalar, const N: usize> Value<T, N> {
    fn chain(
        &self,
        value: T,                            // f
        grad: T,                             // df/dx
        #[cfg(feature = "hessian")] hess: T, // ddf/dxx
    ) -> Self
    where
        T: Zero + One + AddAssign + MulAssign,
    {
        #[cfg(not(feature = "hessian"))]
        return Self {
            value,
            grad: &self.grad * grad,
        };

        #[cfg(feature = "hessian")]
        return Self {
            value,
            grad: &self.grad * grad.clone(),
            hess: &self.grad * self.grad.transpose() * hess + &self.hess * grad,
        };
    }
}
