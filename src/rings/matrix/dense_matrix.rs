use std::alloc::{Allocator, Global};

use crate::{algorithms::matmul::{MatmulAlgorithm, StrassenAlgorithm, STANDARD_MATMUL}, homomorphism::{CanHomFrom, CanIsoFromTo}, integer::{IntegerRing, IntegerRingStore}, matrix::{format_matrix, matrix_add_assign, matrix_negate_inplace, AsFirstElement, OwnedMatrix, Submatrix, SubmatrixMut, TransposableSubmatrix, TransposableSubmatrixMut}, ring::{self, El, EnvBindingStrength, RingBase, RingStore, RingValue}};

use std::fmt::Debug;

///
/// The full matrix ring `M_d(R)`, where R is a commutative ring and d is the dimension of the square matrices.
/// 
/// TODO(bence): comments
/// 
/// TODO(bence): add CanHomFrom R to M_d(R)
/// 
/// TODO(bence): add CanIsoFromTo between M_d(R[x]/(P(x))) and (M_d(R)[X])/P(X), where P(x) is possibly eq to 0
/// 
pub struct DenseMatrixRingBase<R: RingStore, A: Allocator + Clone = Global, M: MatmulAlgorithm<R::Type> = StrassenAlgorithm<Global>> {
    base_ring: R,
    dimension: usize,
    allocator: A,
    matmul_algorithm: M,
}

impl<R: RingStore + Clone, A: Allocator + Clone, M: MatmulAlgorithm<R::Type> + Clone> Clone for DenseMatrixRingBase<R, A, M> {
    fn clone(&self) -> Self {
        DenseMatrixRingBase {
            base_ring: <R as Clone>::clone(&self.base_ring),
            dimension: self.dimension,
            allocator: self.allocator.clone(),
            matmul_algorithm: self.matmul_algorithm.clone()
        }
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> Debug for DenseMatrixRingBase<R, A, M>
    where R::Type: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseMatrixRing")
            .field("base_ring", &self.base_ring.get_ring())
            .field("dimension", &self.dimension)
            .finish()
    }
}

pub type DenseMatrixRing<R, A = Global, M = StrassenAlgorithm> = RingValue<DenseMatrixRingBase<R, A, M>>;

impl<R: RingStore> DenseMatrixRing<R> {

    pub fn new(base_ring: R, dimension: usize) -> Self {
        Self::new_with_matmul(base_ring, dimension, Global, STANDARD_MATMUL)
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> DenseMatrixRing<R, A, M> {

    pub fn new_with_matmul(base_ring: R, dimension: usize, allocator: A, matmul_algorithm: M) -> Self {
        debug_assert!(base_ring.is_commutative());
        RingValue::from(DenseMatrixRingBase {
            base_ring,
            dimension,
            allocator,
            matmul_algorithm
        })
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> DenseMatrixRingBase<R, A, M> {

    pub fn into_base_ring(self) -> R {
        self.base_ring
    }

    pub fn base_ring(&self) -> &R {
        &self.base_ring
    }

    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

///
/// An element of [`DenseMatrixRing`].
/// 
pub struct DenseMatrixRingEl<R: RingStore, A: Allocator + Clone = Global> {
    data: OwnedMatrix<El<R>, A>
}

impl<R, A> Debug for DenseMatrixRingEl<R, A>
    where R: RingStore,
        A: Allocator + Clone,
        El<R>: Debug,
        R: Debug
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.data.fmt(f)
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> RingBase for DenseMatrixRingBase<R, A, M> {

    type Element = DenseMatrixRingEl<R, A>;

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
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                self.base_ring.sub_assign_ref(lhs.data.at_mut(i, j), rhs.data.at(i, j));
            }
        }
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

    fn one(&self) -> Self::Element {
        Self::Element {
            data: OwnedMatrix::identity_in(
                self.dimension, self.dimension, &self.base_ring, self.allocator.clone())
        }
    }

    fn eq_el(&self, lhs: &Self::Element, rhs: &Self::Element) -> bool {
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                if !self.base_ring.eq_el(&lhs.data.at(i, j), &rhs.data.at(i, j)) {
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
        if env >= EnvBindingStrength::Product {
            write!(out, "(")?;
        }
        
        let formatted = format_matrix(
            self.dimension, 
            self.dimension, 
            |i, j| &value.data.at(i, j), 
            &self.base_ring
        );
        write!(out, "{}", formatted)?;
        
        if env >= EnvBindingStrength::Product {
            write!(out, ")")?;
        }
        
        Ok(())
    }

    fn dbg<'a>(&self, value: &Self::Element, out: &mut std::fmt::Formatter<'a>) -> std::fmt::Result {
        self.dbg_within(value, out, EnvBindingStrength::Weakest)
    }

    fn square(&self, value: &mut Self::Element) {
        *value = self.mul_ref(value, value);
    }

    fn mul_ref(&self, lhs: &Self::Element, rhs: &Self::Element) -> Self::Element {
        let mut result = self.zero();
        self.matmul_algorithm.matmul(
            TransposableSubmatrix::from(lhs.data.data()),
            TransposableSubmatrix::from(rhs.data.data()),
            TransposableSubmatrixMut::from(result.data.data_mut()),
            &self.base_ring
        );
        result
    }

    fn mul_assign_int(&self, lhs: &mut Self::Element, rhs: i32) {
        let scalar = self.base_ring.get_ring().from_int(rhs);
        for i in 0..self.dimension {
            for j in 0..self.dimension {
                self.base_ring.mul_assign_ref(&mut lhs.data.at_mut(i, j), &scalar);
            }
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
        let mut iter = els.into_iter();
        if let Some(first) = iter.next() {
            let mut result = first;
            for el in iter {
                result = self.mul_ref(&result, &el);
            }
            result
        } else {
            self.one()
        }
    }

    fn is_approximate(&self) -> bool {
        self.base_ring.get_ring().is_approximate()
    }
}

impl<R, A, M> PartialEq for DenseMatrixRingBase<R, A, M> 
    where R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.dimension == other.dimension && self.base_ring.get_ring() == other.base_ring.get_ring()
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> crate::ring::RingExtension for DenseMatrixRingBase<R, A, M> {
    type BaseRing = R;
    
    fn base_ring(&self) -> &Self::BaseRing {
        &self.base_ring
    }
    
    fn from(&self, x: El<Self::BaseRing>) -> Self::Element {
        Self::Element {
            data: OwnedMatrix::scalar_in(
                self.dimension, self.dimension, x, &self.base_ring, self.allocator.clone())
        }
    }
}

// TODO(bence): implement CanHomFrom R to M_d(R)
// This is blocked by coherence issues: a blanket impl `CanHomFrom<S> for DenseMatrixRingBase<R, ..> where R::Type: CanHomFrom<S>`
// would overlap with the `CanHomFrom<DenseMatrixRingBase>` impl below when `S` is itself a `DenseMatrixRingBase`.

// TODO(bence): implement CanIsoFromTo between M_d(R[x]/(P(x))) and (M_d(R)[X])/P(X), where P(x) is possibly eq to 0

impl<R1: RingStore, R2: RingStore, A1: Allocator + Clone, A2: Allocator + Clone, M1: MatmulAlgorithm<R1::Type>, M2: MatmulAlgorithm<R2::Type>>
    CanHomFrom<DenseMatrixRingBase<R1, A1, M1>> for DenseMatrixRingBase<R2, A2, M2>
    where R2::Type: CanHomFrom<R1::Type>
{
    type Homomorphism = <R2::Type as CanHomFrom<R1::Type>>::Homomorphism;

    fn has_canonical_hom(&self, from: &DenseMatrixRingBase<R1, A1, M1>) -> Option<Self::Homomorphism> {
        if self.dimension != from.dimension {
            return None;
        }
        self.base_ring.get_ring().has_canonical_hom(from.base_ring.get_ring())
    }

    fn map_in(&self, from: &DenseMatrixRingBase<R1, A1, M1>, el: DenseMatrixRingEl<R1, A1>, hom: &Self::Homomorphism) -> Self::Element {
        self.map_in_ref(from, &el, hom)
    }

    fn map_in_ref(&self, from: &DenseMatrixRingBase<R1, A1, M1>, el: &DenseMatrixRingEl<R1, A1>, hom: &Self::Homomorphism) -> Self::Element {
        let d = self.dimension;
        DenseMatrixRingEl {
            data: OwnedMatrix::from_fn_in(d, d, |i, j| {
                self.base_ring.get_ring().map_in_ref(from.base_ring.get_ring(), &el.data.at(i, j), hom)
            }, self.allocator.clone())
        }
    }
}

impl<R1: RingStore, R2: RingStore, A1: Allocator + Clone, A2: Allocator + Clone, M1: MatmulAlgorithm<R1::Type>, M2: MatmulAlgorithm<R2::Type>>
    CanIsoFromTo<DenseMatrixRingBase<R1, A1, M1>> for DenseMatrixRingBase<R2, A2, M2>
    where R2::Type: CanIsoFromTo<R1::Type>
{
    type Isomorphism = <R2::Type as CanIsoFromTo<R1::Type>>::Isomorphism;

    fn has_canonical_iso(&self, from: &DenseMatrixRingBase<R1, A1, M1>) -> Option<Self::Isomorphism> {
        if self.dimension != from.dimension {
            return None;
        }
        self.base_ring.get_ring().has_canonical_iso(from.base_ring.get_ring())
    }

    fn map_out(&self, from: &DenseMatrixRingBase<R1, A1, M1>, el: Self::Element, iso: &Self::Isomorphism) -> <DenseMatrixRingBase<R1, A1, M1> as RingBase>::Element {
        let d = self.dimension;
        DenseMatrixRingEl {
            data: OwnedMatrix::from_fn_in(d, d, |i, j| {
                self.base_ring.get_ring().map_out(from.base_ring.get_ring(), self.base_ring.clone_el(&el.data.at(i, j)), iso)
            }, from.allocator.clone())
        }
    }
}

impl<R: RingStore, A: Allocator + Clone, M: MatmulAlgorithm<R::Type>> crate::rings::matrix::MatrixRing for DenseMatrixRingBase<R, A, M> {

    fn to_elements<'a>(&'a self, el: &'a <DenseMatrixRingBase<R, A, M> as ring::RingBase>::Element) -> Submatrix<'a, AsFirstElement<El<R>>, El<R>>
    {
        return el.data.data();
    }

    fn to_elements_mut<'a>(&'a self, el: &'a mut <DenseMatrixRingBase<R, A, M> as ring::RingBase>::Element) -> SubmatrixMut<'a, AsFirstElement<El<R>>, El<R>>
    {
        return el.data.data_mut();
    }

    fn from_elements(&self, elements: Vec<El<Self::BaseRing>>) -> Self::Element {
        assert_eq!(elements.len(), self.dimension * self.dimension, 
                  "Vector must have exactly dimension^2 elements");

        let mut data = Vec::with_capacity_in(self.dimension * self.dimension, self.allocator.clone());
        for element in elements {
            data.push(element);
        }

        Self::Element {
            data: OwnedMatrix::new_with_shape(data, self.dimension, self.dimension)
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn element_at<'a>(&self, el: &'a Self::Element, i: usize, j: usize) -> &'a El<Self::BaseRing> {
        &el.data.at(i, j)
    }

    fn set_element_at(&self, el: &mut Self::Element, i: usize, j: usize, value: El<Self::BaseRing>) {
        *el.data.at_mut(i, j) = value;
    }
}

#[cfg(test)]
use crate::primitive_int::StaticRing;
#[cfg(test)]
use crate::rings::zn::zn_64::Zn;
#[cfg(test)]
use crate::ring::RingExtensionStore;
#[cfg(test)]
use crate::homomorphism::Homomorphism;
#[cfg(test)]
use crate::rings::matrix::MatrixRing;

#[cfg(test)]
fn edge_case_elements_2x2<R: RingStore>(ring: &DenseMatrixRing<R>) -> Vec<El<DenseMatrixRing<R>>>
    where R: Clone
{
    let base = ring.base_ring();
    let from_arr = |arr: [[i32; 2]; 2]| -> El<DenseMatrixRing<R>> {
        ring.get_ring().from_elements(vec![
            base.int_hom().map(arr[0][0]),
            base.int_hom().map(arr[0][1]),
            base.int_hom().map(arr[1][0]),
            base.int_hom().map(arr[1][1]),
        ])
    };
    vec![
        ring.zero(),
        ring.one(),
        ring.int_hom().map(-1),
        from_arr([[1, 2], [3, 4]]),
        from_arr([[0, 1], [0, 0]]),
        from_arr([[0, 0], [1, 0]]),
        from_arr([[1, 1], [0, 1]]),
        from_arr([[2, 0], [0, -1]]),
    ]
}

///
/// The generic `test_ring_axioms` incorrectly checks `(a*b)*c == c*(b*a)` for
/// multiplicative associativity, which is only valid in commutative rings.
/// Matrix rings are non-commutative, so we test ring axioms manually.
///
#[test]
fn test_ring_axioms() {
    let ring = DenseMatrixRing::new(StaticRing::<i64>::RING, 2);
    test_matrix_ring_axioms(&ring, &edge_case_elements_2x2(&ring));
}

#[test]
fn test_ring_axioms_zn() {
    let ring = DenseMatrixRing::new(Zn::new(7), 2);
    test_matrix_ring_axioms(&ring, &edge_case_elements_2x2(&ring));
}

#[cfg(test)]
fn test_matrix_ring_axioms<R: RingStore + Clone>(ring: &DenseMatrixRing<R>, elements: &[El<DenseMatrixRing<R>>]) {
    use crate::ring::EnvBindingStrength;

    let zero = ring.zero();
    let one = ring.one();

    assert!(ring.is_zero(&zero));
    assert!(ring.is_one(&one));

    // identity and inverse
    for a in elements {
        let a_minus_a = ring.sub(ring.clone_el(a), ring.clone_el(a));
        assert!(ring.eq_el(&zero, &a_minus_a), "Additive inverse failed: {} - {} = {} != 0", ring.format(a), ring.format(a), ring.format(&a_minus_a));

        let a_plus_zero = ring.add(ring.clone_el(a), ring.clone_el(&zero));
        assert!(ring.eq_el(a, &a_plus_zero), "Additive identity failed");

        let a_times_one = ring.mul(ring.clone_el(a), ring.clone_el(&one));
        assert!(ring.eq_el(a, &a_times_one), "Multiplicative identity (right) failed");

        let one_times_a = ring.mul(ring.clone_el(&one), ring.clone_el(a));
        assert!(ring.eq_el(a, &one_times_a), "Multiplicative identity (left) failed");
    }

    // commutativity of addition
    for a in elements {
        for b in elements {
            let ab = ring.add_ref(a, b);
            let ba = ring.add_ref(b, a);
            assert!(ring.eq_el(&ab, &ba), "Additive commutativity failed");
        }
    }

    // associativity
    for a in elements {
        for b in elements {
            for c in elements {
                {
                    let ab_c = ring.add(ring.add_ref(a, b), ring.clone_el(c));
                    let a_bc = ring.add(ring.clone_el(a), ring.add_ref(b, c));
                    assert!(ring.eq_el(&ab_c, &a_bc), "Additive associativity failed");
                }
                {
                    let ab_c = ring.mul(ring.mul_ref(a, b), ring.clone_el(c));
                    let a_bc = ring.mul(ring.clone_el(a), ring.mul_ref(b, c));
                    assert!(ring.eq_el(&ab_c, &a_bc), "Multiplicative associativity failed: ({} * {}) * {} = {} != {} = {} * ({} * {})",
                        ring.format_within(a, EnvBindingStrength::Product),
                        ring.format_within(b, EnvBindingStrength::Product),
                        ring.format_within(c, EnvBindingStrength::Product),
                        ring.format(&ab_c),
                        ring.format(&a_bc),
                        ring.format_within(a, EnvBindingStrength::Product),
                        ring.format_within(b, EnvBindingStrength::Product),
                        ring.format_within(c, EnvBindingStrength::Product));
                }
            }
        }
    }

    // distributivity
    for a in elements {
        for b in elements {
            for c in elements {
                let ab_c = ring.mul(ring.add_ref(a, b), ring.clone_el(c));
                let ac_bc = ring.add(ring.mul_ref(a, c), ring.mul_ref(b, c));
                assert!(ring.eq_el(&ab_c, &ac_bc), "Right distributivity failed");

                let a_bc = ring.mul(ring.clone_el(a), ring.add_ref(b, c));
                let ab_ac = ring.add(ring.mul_ref(a, b), ring.mul_ref(a, c));
                assert!(ring.eq_el(&a_bc, &ab_ac), "Left distributivity failed");
            }
        }
    }
}

#[test]
fn test_self_iso() {
    let ring = DenseMatrixRing::new(Zn::new(7), 2);
    crate::ring::generic_tests::test_self_iso(&ring, edge_case_elements_2x2(&ring).into_iter());
}

#[test]
fn test_hom_axioms() {
    let from = DenseMatrixRing::new(StaticRing::<i64>::RING, 2);
    let to = DenseMatrixRing::new(Zn::new(7), 2);
    crate::ring::generic_tests::test_hom_axioms(&from, &to, edge_case_elements_2x2(&from).into_iter());
}

#[test]
fn test_iso_axioms() {
    let from = DenseMatrixRing::new(Zn::new(7), 2);
    let to = DenseMatrixRing::new(Zn::new(7), 2);
    crate::ring::generic_tests::test_iso_axioms(&from, &to, edge_case_elements_2x2(&from).into_iter());
}

#[test]
fn test_dimension_mismatch() {
    let ring_2x2 = DenseMatrixRing::new(Zn::new(7), 2);
    let ring_3x3 = DenseMatrixRing::new(Zn::new(7), 3);
    assert_ne!(ring_2x2.get_ring(), ring_3x3.get_ring());
    assert!(ring_3x3.get_ring().has_canonical_hom(ring_2x2.get_ring()).is_none());
    assert!(ring_3x3.get_ring().has_canonical_iso(ring_2x2.get_ring()).is_none());
}

#[test]
fn test_matrix_mul() {
    let ring = DenseMatrixRing::new(StaticRing::<i64>::RING, 2);
    let base = ring.base_ring();
    // [[1, 2], [3, 4]] * [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
    let a = ring.get_ring().from_elements(vec![
        base.int_hom().map(1), base.int_hom().map(2),
        base.int_hom().map(3), base.int_hom().map(4),
    ]);
    let b = ring.get_ring().from_elements(vec![
        base.int_hom().map(5), base.int_hom().map(6),
        base.int_hom().map(7), base.int_hom().map(8),
    ]);
    let expected = ring.get_ring().from_elements(vec![
        base.int_hom().map(19), base.int_hom().map(22),
        base.int_hom().map(43), base.int_hom().map(50),
    ]);
    let actual = ring.mul(a, b);
    assert!(ring.eq_el(&expected, &actual));
}

#[test]
fn test_poly_ring_over_matrix_ring() {
    use crate::rings::poly::dense_poly::DensePolyRing;
    use crate::rings::poly::PolyRingStore;

    let ZZ = StaticRing::<i64>::RING;
    let m2 = DenseMatrixRing::new(ZZ, 2);

    // M_2(Z)[X]
    let poly_ring = DensePolyRing::new(m2.clone(), "X");
    let x = poly_ring.indeterminate();

    // create matrix coefficients: A = [[1, 2], [3, 4]], B = [[0, 1], [1, 0]]
    let mat_a = m2.get_ring().from_elements(vec![1, 2, 3, 4]);
    let mat_b = m2.get_ring().from_elements(vec![0, 1, 1, 0]);

    // f(X) = A + B*X
    let f = poly_ring.add(
        poly_ring.inclusion().map(m2.clone_el(&mat_a)),
        poly_ring.mul(poly_ring.inclusion().map(m2.clone_el(&mat_b)), poly_ring.clone_el(&x)),
    );
    assert_eq!(Some(1), poly_ring.degree(&f));
    assert!(m2.eq_el(&mat_a, poly_ring.coefficient_at(&f, 0)));
    assert!(m2.eq_el(&mat_b, poly_ring.coefficient_at(&f, 1)));

    // g(X) = I + X  (identity matrix + X)
    let g = poly_ring.add(
        poly_ring.inclusion().map(m2.one()),
        poly_ring.clone_el(&x),
    );
    assert_eq!(Some(1), poly_ring.degree(&g));
    assert!(m2.eq_el(&m2.one(), poly_ring.coefficient_at(&g, 0)));

    // f * g = (A + B*X) * (I + X)
    // convolution gives c_i = sum_j f[i-j] * g[j]
    // c_0 = A*I = A
    // c_1 = B*I + A*I = A + B
    // c_2 = B*I = B
    let fg = poly_ring.mul_ref(&f, &g);
    assert_eq!(Some(2), poly_ring.degree(&fg));
    let expected_fg_c1 = m2.add(m2.clone_el(&mat_a), m2.clone_el(&mat_b));
    assert!(m2.eq_el(&mat_a, poly_ring.coefficient_at(&fg, 0)));
    assert!(m2.eq_el(&expected_fg_c1, poly_ring.coefficient_at(&fg, 1)));
    assert!(m2.eq_el(&mat_b, poly_ring.coefficient_at(&fg, 2)));

    // f^2 = (A + B*X)^2
    // c_0 = A*A, c_1 = B*A + A*B, c_2 = B*B
    let f_sq = poly_ring.mul_ref(&f, &f);
    assert_eq!(Some(2), poly_ring.degree(&f_sq));
    let expected_sq_c0 = m2.mul_ref(&mat_a, &mat_a);
    let expected_sq_c1 = m2.add(m2.mul_ref(&mat_b, &mat_a), m2.mul_ref(&mat_a, &mat_b));
    let expected_sq_c2 = m2.mul_ref(&mat_b, &mat_b);
    assert!(m2.eq_el(&expected_sq_c0, poly_ring.coefficient_at(&f_sq, 0)));
    assert!(m2.eq_el(&expected_sq_c1, poly_ring.coefficient_at(&f_sq, 1)));
    assert!(m2.eq_el(&expected_sq_c2, poly_ring.coefficient_at(&f_sq, 2)));

    // test X^3
    let x_cubed = poly_ring.pow(poly_ring.clone_el(&x), 3);
    assert_eq!(Some(3), poly_ring.degree(&x_cubed));
    assert!(m2.is_zero(poly_ring.coefficient_at(&x_cubed, 0)));
    assert!(m2.is_zero(poly_ring.coefficient_at(&x_cubed, 1)));
    assert!(m2.is_zero(poly_ring.coefficient_at(&x_cubed, 2)));
    assert!(m2.is_one(poly_ring.coefficient_at(&x_cubed, 3)));

    // test zero polynomial
    let zero = poly_ring.zero();
    assert_eq!(None, poly_ring.degree(&zero));
    assert!(poly_ring.is_zero(&poly_ring.mul(poly_ring.clone_el(&zero), poly_ring.clone_el(&f))));
}