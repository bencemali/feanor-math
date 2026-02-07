use crate::{matrix::{AsFirstElement, Submatrix, SubmatrixMut}, ring::{self, El, RingExtension, RingStore}};

pub mod dense_matrix;

// TODO(bence): implement DiagonalMatrixRing
pub mod diagonal_matrix;

// TODO(bence): implement SparseMatrixRing
pub mod sparse_matrix;

///
/// Trait for all rings that represent the square matrix ring `M_d(R)`, where `R` is a commutative ring and `d` is the dimension of the square matrices.
///
pub trait MatrixRing: RingExtension {

    // type ElementsView<'a> = Submatrix<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>
    //     where Self: 'a;

    // type ElementsViewMut<'a> = SubmatrixMut<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>
    //     where Self: 'a;

    ///
    /// Returns a view of the elements of the given matrix.
    ///
    fn to_elements<'a>(&'a self, el: &'a Self::Element) -> Submatrix<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>;

    ///
    /// Returns a mutable view of the elements of the given matrix.
    ///
    fn to_elements_mut<'a>(&'a self, el: &'a mut Self::Element) -> SubmatrixMut<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>;

    ///
    /// Creates a matrix from the given elements.
    ///
    fn from_elements(&self, elements: Vec<El<Self::BaseRing>>) -> Self::Element;

    ///
    /// Returns the dimension of the given matrix.
    ///
    fn dimension(&self) -> usize;

    ///
    /// Returns the element at the given position in the given matrix.
    ///
    fn element_at<'a>(&self, el: &'a Self::Element, i: usize, j: usize) -> &'a El<Self::BaseRing>;

    ///
    /// Sets the element at the given position in the given matrix.
    ///
    fn set_element_at(&self, el: &mut Self::Element, i: usize, j: usize, value: El<Self::BaseRing>);
}

pub trait MatrixRingStore: RingStore
    where Self::Type: MatrixRing
{
    /// See [`MatrixRing::to_elements()`]
    fn to_elements<'a>(&'a self, el: &'a El<Self>) -> Submatrix<'a, AsFirstElement<El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>>, El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>> {
        <Self::Type as MatrixRing>::to_elements(self.get_ring(), el)
    }

    /// See [`MatrixRing::to_elements_mut()`]
    fn to_elements_mut<'a>(&'a self, el: &'a mut El<Self>) -> SubmatrixMut<'a, AsFirstElement<El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>>, El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>> {
        <Self::Type as MatrixRing>::to_elements_mut(self.get_ring(), el)
    }

    delegate!{ MatrixRing, fn from_elements(&self, elements: Vec<El<<Self::Type as RingExtension>::BaseRing>>) -> El<Self> }
    delegate!{ MatrixRing, fn dimension(&self) -> usize }

    /// See [`MatrixRing::element_at()`]
    fn element_at<'a>(&self, el: &'a El<Self>, i: usize, j: usize) -> &'a El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing> {
        <Self::Type as MatrixRing>::element_at(self.get_ring(), el, i, j)
    }

    /// See [`MatrixRing::set_element_at()`]
    fn set_element_at(&self, el: &mut El<Self>, i: usize, j: usize, value: El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>) {
        <Self::Type as MatrixRing>::set_element_at(self.get_ring(), el, i, j, value)
    }
}

// TODO(bence): tests