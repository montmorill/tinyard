use std::ops::{Add, Div, Mul, Neg, Rem, Sub};
use std::ops::{AddAssign, DivAssign, MulAssign, RemAssign, SubAssign};

use nalgebra::{ClosedAddAssign, ClosedDivAssign, ClosedSubAssign};
use num_traits::{ConstOne, ConstZero, Inv, Num, NumAssign, One, Signed, Zero};
use simba::scalar::ClosedNeg;

use crate::Value;

macro_rules! binary_op_impl {
    (@first $Trait:ident $method:ident $TraitAssign:ident $method_assign:ident) => {
        impl<T: nalgebra::Scalar, const N: usize> $Trait for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign
        {
            type Output = Self;

            #[inline]
            fn $method(mut self, rhs: Self) -> Self::Output {
                self.$method_assign(rhs);
                self
            }
        }

        impl<T: nalgebra::Scalar, const N: usize> $Trait<&Self> for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign
        {
            type Output = Self;

            #[inline]
            fn $method(mut self, rhs: &Self) -> Self::Output {
                self.$method_assign(rhs);
                self
            }
        }

        binary_op_impl!(@swap $Trait $method $TraitAssign $method_assign);
    };

    (@second $Trait:ident $method:ident $TraitAssign:ident $method_assign:ident $($($Bound:tt)+)?) => {
        impl<T: nalgebra::Scalar, const N: usize> $Trait<&Self> for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($Bound)+)?
        {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: &Self) -> Self::Output {
                self.$method(rhs.clone())
            }
        }

        binary_op_impl!(@swap $Trait $method $TraitAssign $method_assign $($($Bound)+)?);

        impl<T: nalgebra::Scalar, const N: usize> $TraitAssign for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($Bound)+)?
        {
            #[inline]
            fn $method_assign(&mut self, rhs: Self) {
                *self = (&*self).$method(rhs);
            }
        }

        impl<T: nalgebra::Scalar, const N: usize> $TraitAssign<&Self> for Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($Bound)+)?
        {
            #[inline]
            fn $method_assign(&mut self, rhs: &Self) {
                self.$method_assign(rhs.clone());
            }
        }
    };

    (@swap $Trait:ident $method:ident $TraitAssign:ident $method_assign:ident $($($Bound:tt)+)?) => {
        impl<T: nalgebra::Scalar, const N: usize> $Trait<Value<T, N>> for &Value<T, N>
        where
            T: $Trait<Output = T> + $TraitAssign $(+ $($Bound)+)?
        {
            type Output = Value<T, N>;

            #[inline]
            fn $method(self, rhs: Value<T, N>) -> Self::Output {
                rhs.$method(self)
            }
        }
    }
}

impl<'a, T: nalgebra::Scalar, const N: usize> AddAssign for Value<T, N>
where
    T: ClosedAddAssign,
{
    fn add_assign(&mut self, rhs: Self) {
        self.value += rhs.value;
        self.grad += rhs.grad;
        #[cfg(feature = "hessian")]
        {
            self.hess += rhs.hess;
        }
    }
}

impl<T: nalgebra::Scalar, const N: usize> AddAssign<&Self> for Value<T, N>
where
    T: ClosedAddAssign,
{
    fn add_assign(&mut self, rhs: &Self) {
        self.value += rhs.value.clone();
        self.grad += &rhs.grad;
        #[cfg(feature = "hessian")]
        {
            self.hess += &rhs.hess;
        }
    }
}

binary_op_impl!(@first Add add AddAssign add_assign);
impl<T: nalgebra::Scalar, const N: usize> SubAssign for Value<T, N>
where
    T: ClosedSubAssign,
{
    fn sub_assign(&mut self, rhs: Self) {
        self.value -= rhs.value;
        self.grad -= rhs.grad;
        #[cfg(feature = "hessian")]
        {
            self.hess -= rhs.hess;
        }
    }
}

impl<T: nalgebra::Scalar, const N: usize> SubAssign<&Self> for Value<T, N>
where
    T: ClosedSubAssign,
{
    fn sub_assign(&mut self, rhs: &Self) {
        self.value -= rhs.value.clone();
        self.grad -= &rhs.grad;
        #[cfg(feature = "hessian")]
        {
            self.hess -= &rhs.hess;
        }
    }
}

binary_op_impl!(@first Sub sub SubAssign sub_assign);

impl<T: nalgebra::Scalar, const N: usize> Mul for Value<T, N>
where
    // `Zero` and `One` imply a `Closed` verison of `AddAssign` and `MulAssign`.
    T: AddAssign + MulAssign + Zero + One,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            value: self.value.clone() * rhs.value.clone(),
            #[cfg(feature = "hessian")]
            hess: self.hess * rhs.value.clone()
                + self.grad.clone() * rhs.grad.transpose()
                + rhs.grad.clone() * self.grad.transpose()
                + rhs.hess * self.value.clone(),
            grad: self.grad * rhs.value + rhs.grad * self.value,
        }
    }
}

binary_op_impl!(@second Mul mul MulAssign mul_assign AddAssign + Zero + One);

impl<T: nalgebra::Scalar, const N: usize> Div for Value<T, N>
where
    // `Zero` and `One` imply a `Closed` verison of `AddAssign` and `MulAssign`.
    T: AddAssign + ClosedSubAssign + MulAssign + ClosedDivAssign + Zero + One,
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let value = self.value.clone() / rhs.value.clone();
        #[cfg(not(feature = "hessian"))]
        let grad = (self.grad * rhs.value.clone() - rhs.grad * self.value)
            / (rhs.value.clone() * rhs.value);
        #[cfg(feature = "hessian")]
        let grad = (self.grad * rhs.value.clone() - rhs.grad.clone() * self.value)
            / (rhs.value.clone() * rhs.value.clone());
        Value {
            #[cfg(feature = "hessian")]
            hess: (self.hess
                - grad.clone() * rhs.grad.transpose()
                - rhs.grad * grad.transpose()
                - rhs.hess * value.clone())
                / rhs.value,
            value,
            grad,
        }
    }
}

binary_op_impl!(@second Div div DivAssign div_assign AddAssign + ClosedSubAssign + MulAssign + Zero + One);

impl<T: nalgebra::Scalar, const N: usize> RemAssign for Value<T, N>
where
    T: RemAssign,
{
    fn rem_assign(&mut self, rhs: Self) {
        self.value %= rhs.value;
    }
}

impl<T: nalgebra::Scalar, const N: usize> RemAssign<&Self> for Value<T, N>
where
    T: RemAssign,
{
    fn rem_assign(&mut self, rhs: &Self) {
        self.value %= rhs.value.clone();
    }
}

binary_op_impl!(@first Rem rem RemAssign rem_assign);

impl<T: nalgebra::Scalar, const N: usize> Zero for Value<T, N>
where
    T: AddAssign + Zero,
{
    fn zero() -> Self {
        Self::new(T::zero())
    }

    fn is_zero(&self) -> bool {
        self.value.is_zero()
            && self.grad.as_slice().into_iter().all(T::is_zero)
            && (|| {
                #[cfg(feature = "hessian")]
                return self.hess.as_slice().into_iter().all(T::is_zero);
                #[allow(unreachable_code)]
                true
            })()
    }
}

impl<T: nalgebra::Scalar, const N: usize> ConstZero for Value<T, N>
where
    T: AddAssign + ConstZero + Copy,
{
    const ZERO: Self = Self::new_const(T::ZERO);
}

impl<T: nalgebra::Scalar, const N: usize> One for Value<T, N>
where
    T: AddAssign + MulAssign + Zero + One,
{
    fn one() -> Self {
        Self::new(T::one())
    }
}

impl<T: nalgebra::Scalar, const N: usize> ConstOne for Value<T, N>
where
    T: AddAssign + MulAssign + ConstZero + ConstOne + Copy,
{
    const ONE: Self = Self::new_const(T::ONE);
}

impl<T: nalgebra::Scalar + NumAssign, const N: usize> Num for Value<T, N> {
    type FromStrRadixErr = T::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        T::from_str_radix(str, radix).map(|value| Self::new(value))
    }
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

impl<T: nalgebra::Scalar, const N: usize> Inv for Value<T, N>
where
    Self: Div<Output = Value<T, N>> + One,
{
    type Output = Self;

    fn inv(self) -> Self::Output {
        Self::one() / self
    }
}

impl<T: nalgebra::Scalar + NumAssign, const N: usize> Signed for Value<T, N>
where
    T: Signed,
{
    fn abs(&self) -> Self {
        if self.value.is_negative() {
            -self.clone()
        } else {
            self.clone()
        }
    }

    fn abs_sub(&self, other: &Self) -> Self {
        let result = self.sub(other.clone());
        if result.is_positive() {
            result
        } else {
            Zero::zero()
        }
    }

    fn signum(&self) -> Self {
        Self::new(self.value.signum())
    }

    fn is_positive(&self) -> bool {
        self.value.is_positive()
    }

    fn is_negative(&self) -> bool {
        self.value.is_negative()
    }
}

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
