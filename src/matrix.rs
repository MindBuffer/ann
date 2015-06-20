

pub use ::nalgebra::DMat as Mat;



/// Update each element in the matrix by setting each element with the result of the given function.
pub fn update_elems<T, F>(mat: &mut Mat<T>, mut f: F) where
    T: Copy,
    F: FnMut(T) -> T,
{
    for elem in mat.as_mut_vec().iter_mut() {
        *elem = f(*elem);
    }
}


/// Normalise each column of a matrix.
pub fn normalise_cols<N>(mat: &mut Mat<N>) where
    N: Copy + PartialOrd + ::std::ops::Div<N, Output=N> + ::num::traits::Zero,
{
    let num_rows = mat.nrows();
    for col in mat.as_mut_vec().chunks_mut(num_rows) {
        let max = col.iter().fold(N::zero(), |max, &elem| {
            if elem > max { elem } else { max }
        });
        // If the max value is zero we can't normalise so continue to next column.
        if max == N::zero() { continue; }
        for elem in col.iter_mut() {
            *elem = *elem / max;
        }
    }
}
