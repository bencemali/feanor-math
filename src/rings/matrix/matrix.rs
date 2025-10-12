use std::alloc::{Allocator, Global};

use crate::{algorithms::matmul::{MatmulAlgorithm, StrassenAlgorithm, STANDARD_MATMUL}, integer::{IntegerRing, IntegerRingStore}, matrix::{matrix_add_assign, matrix_negate_inplace, matrix_sub_self_assign, AsFirstElement, AsPointerToSlice, OwnedMatrix, TransposableSubmatrix, TransposableSubmatrixMut}, ring::{EnvBindingStrength, RingBase, RingRef, RingStore, RingValue}};
use crate::ring::El;

use std::fmt::Debug;

///
/// The full matrix ring `M_d(R)`, where R is a commutative ring and d is the dimension of the square matrices.
/// 
/// TODO(bence): comments
/// 
/// TODO(bence): add CanIsoFromTo between M_d(R[x]/(P(x))) and (M_d(R)[X])/P(X), where P(x) is possibly eq to 0
/// 
/// TODO(bence): impl RingExtension/CanHomFrom (base_ring -> MatrixRing) for MatrixRingBase
/// 
pub struct MatrixRingBase<R: RingStore, A: Allocator + Clone = Global, M: MatmulAlgorithm<R::Type> = StrassenAlgorithm<Global>> {
    base_ring: R,
    dimension: usize,
    allocator: A,
    matmul_algorithm: M,
}

impl<R: RingStore + Clone, A: Allocator + Clone, M: MatmulAlgorithm<R::Type> + Clone> Clone for MatrixRingBase<R, A, M> {

    fn clone(&self) -> Self {
        MatrixRingBase {
            base_ring: <R as Clone>::clone(&self.base_ring),
            dimension: self.dimension,
            allocator: self.allocator.clone(),
            matmul_algorithm: self.matmul_algorithm.clone()
        }
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> Debug for MatrixRingBase<R, A, M>
    where R::Type: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixRing")
            .field("base_ring", &self.base_ring.get_ring())
            .field("dimension", &self.dimension)
            .finish()
    }
}

pub type MatrixRing<R, A = Global, M = StrassenAlgorithm> = RingValue<MatrixRingBase<R, A, M>>;

impl<R: RingStore> MatrixRing<R> {

    pub fn new(base_ring: R, dimension: usize) -> Self {
        Self::new_with_matmul(base_ring, dimension, Global, STANDARD_MATMUL)
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> MatrixRing<R, A, M> {

    pub fn new_with_matmul(base_ring: R, dimension: usize, allocator: A, matmul_algorithm: M) -> Self {
        debug_assert!(base_ring.is_commutative());
        RingValue::from(MatrixRingBase {
            base_ring,
            dimension,
            allocator,
            matmul_algorithm
        })
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> MatrixRingBase<R, A, M> {

    pub fn into_base_ring(self) -> R {
        self.base_ring
    }
}

///
/// An element of [`MatrixRing`].
/// 
pub struct MatrixRingEl<R: RingStore, A: Allocator + Clone = Global> {
    // TODO(bence): is OwnedMatrix the most efficient solution?
    // maybe inefficient to always convert to TransposableSubmatrix(Mut)
    data: OwnedMatrix<El<R>, A>
}

impl<R, A> Debug for MatrixRingEl<R, A>
    where R: RingStore,
        A: Allocator + Clone,
        El<R>: Debug,
        R: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<R, A, M> RingBase for MatrixRingBase<R, A, M> 
    where R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>
{

    type Element = MatrixRingEl<R, A>;

    fn clone_el(&self, val: &Self::Element) -> Self::Element {
        Self::Element { data: val.data.clone_matrix(&self.base_ring) }
    }

    fn add_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        matrix_add_assign(TransposableSubmatrix::from(rhs.data.data()), 
            TransposableSubmatrixMut::from(lhs.data.data_mut()), &self.base_ring);
    }

    fn add_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.add_assign_ref(lhs, &rhs);
    }

    fn sub_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        matrix_sub_self_assign(TransposableSubmatrix::from(rhs.data.data()),
            TransposableSubmatrixMut::from(lhs.data.data_mut()), &self.base_ring);
    }

    fn negate_inplace(&self, value: &mut Self::Element) {
        matrix_negate_inplace::<AsFirstElement<El<R>>, R, false>
            (TransposableSubmatrixMut::from(value.data.data_mut()), &self.base_ring);
    }

    fn mul_assign(&self, lhs: &mut Self::Element, rhs: Self::Element) {
        self.mul_assign_ref(lhs, &rhs);
    }

    fn mul_assign_ref(&self, lhs: &mut Self::Element, rhs: &Self::Element) {
        *lhs = self.mul_ref(lhs, rhs);
    }

    fn zero(&self) -> Self::Element {
        Self::Element {
            data: OwnedMatrix::zero_in(
                self.dimension, self.dimension, &self.base_ring, self.allocator.clone())
        }
    }

    fn from_int(&self, value: i32) -> Self::Element {
        Self::Element {
            data: OwnedMatrix::scalar_in(
                self.dimension, self.dimension, self.base_ring.get_ring().from_int(value), &self.base_ring, self.allocator.clone())
        }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        if lhs.data.col_count() != lhs.data.row_count()
            || lhs.data.col_count() != rhs.data.col_count()
            || rhs.data.col_count() != rhs.data.row_count()
        {
            return false;
        }
        let dim = lhs.data.col_count();
        for i in 0..dim {
            for j in 0..dim {
                if !self.base_ring.eq_el(lhs.data.at(i, j), rhs.data.at(i, j))
                {
                    return false;
                }
            }
        }
        return true;
    }

    fn is_commutative(&self) -> bool {
        false
    }

    fn is_noetherian(&self) -> bool {
        // R Noetherian & matrix ring contains all scalar matrices <=> matrix ring Noetherian
        self.base_ring.is_noetherian()
    }

    fn dbg_within<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>, env: EnvBindingStrength) -> std::fmt::Result {
        // TODO(bence)
        unimplemented!("MatrixRingBase::dbg_within")
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        // TODO(bence)
        unimplemented!("MatrixRingBase::dbg")
    }

    fn square(&self, value: &mut Self::Element) {
        *value = self.mul_ref(&value, &value);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let vec = Vec::with_capacity_in(
            self.dimension * self.dimension,
            self.allocator.clone()
        );
        let mut owned = OwnedMatrix::new(vec, self.dimension);
        self.matmul_algorithm.matmul(
            TransposableSubmatrix::from(lhs.data.data()),
            TransposableSubmatrix::from(rhs.data.data()),
            TransposableSubmatrixMut::from(owned.data_mut()),
            &self.base_ring
        );
        Self::Element {
            data: owned
        }
    }

    fn characteristic<I: IntegerRingStore + Copy>(&self, ZZ: I) -> Option<El<I>>
        where I::Type: IntegerRing
    {
        self.base_ring.get_ring().characteristic(ZZ)
    }

    fn prod<I>(&self, els: I) -> Self::Element 
        where I: IntoIterator<Item = Self::Element>
    {
        els.into_iter().fold(self.one(), |a, b| self.mul(a, b))
    }

    fn is_approximate(&self) -> bool {
        self.base_ring.get_ring().is_approximate()
    }
}

impl<R, A, M> PartialEq for MatrixRingBase<R, A, M> 
    where R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

// TODO(bence): tests