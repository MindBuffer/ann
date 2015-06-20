

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

