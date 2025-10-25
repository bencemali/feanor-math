use crate::{matrix::{AsFirstElement, Submatrix, SubmatrixMut}, ring::{self, El, RingExtension, RingStore}};

pub mod matrix;

pub trait MatrixRing: RingExtension {

    // type ElementsView<'a> = Submatrix<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>
    //     where Self: 'a;

    // type ElementsViewMut<'a> = SubmatrixMut<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>
    //     where Self: 'a;

    fn to_elements<'a>(&'a self, el: &'a Self::Element) -> Submatrix<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>;

    fn to_elements_mut<'a>(&'a self, el: &'a mut Self::Element) -> SubmatrixMut<'a, AsFirstElement<El<Self::BaseRing>>, El<Self::BaseRing>>;

    fn from_elements(&self, elements: Vec<El<Self::BaseRing>>) -> Self::Element;

    fn dimension(&self) -> usize;
}

pub trait MatrixRingStore: RingStore
    where Self::Type: MatrixRing
{
    delegate!{ MatrixRing, fn dimension(&self) -> usize }
    delegate!{ MatrixRing, fn from_elements(&self, elements: Vec<El<<Self::Type as RingExtension>::BaseRing>>) -> El<Self> }

    fn elements<'a>(&'a self, el: &'a El<Self>) -> Submatrix<'a, AsFirstElement<El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>>, El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>> {
        self.get_ring().to_elements(el)
    }

    fn elements_mut<'a>(&'a self, el: &'a mut El<Self>) -> SubmatrixMut<'a, AsFirstElement<El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>>, El<<<Self as ring::RingStore>::Type as ring::RingExtension>::BaseRing>> {
        self.get_ring().to_elements_mut(el)
    }
}