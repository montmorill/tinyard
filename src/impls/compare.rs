use approx::{AbsDiffEq, RelativeEq, UlpsEq};

use crate::Tin;

impl<T: nalgebra::Scalar, const N: usize> PartialEq for Tin<T, N> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<T: nalgebra::Scalar, const N: usize> Eq for Tin<T, N> where T: Eq {}

impl<T: nalgebra::Scalar, const N: usize> PartialOrd for Tin<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: nalgebra::Scalar, const N: usize> Ord for Tin<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<T: nalgebra::Scalar, const N: usize> AbsDiffEq for Tin<T, N>
where
    T: AbsDiffEq,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> T::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: T::Epsilon) -> bool {
        self.value.abs_diff_eq(&other.value, epsilon)
    }
}

impl<T: nalgebra::Scalar, const N: usize> RelativeEq for Tin<T, N>
where
    T: RelativeEq,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.value.relative_eq(&other.value, epsilon, max_relative)
    }
}

impl<T: nalgebra::Scalar, const N: usize> UlpsEq for Tin<T, N>
where
    T: UlpsEq,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: T::Epsilon, max_ulps: u32) -> bool {
        self.value.ulps_eq(&other.value, epsilon, max_ulps)
    }
}
