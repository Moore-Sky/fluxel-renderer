//! Tests public headless renderer domain validation and draw ordering.

use super::*;

#[test]
fn geometry_rejects_an_out_of_bounds_index() {
    let error = Geometry::from_positions(vec![[0.0, 0.0, 0.0]])
        .with_indices(vec![1])
        .unwrap_err();
    assert_eq!(
        error,
        GeometryError::IndexOutOfBounds {
            index: 1,
            vertex_count: 1,
        }
    );
}

#[test]
fn draw_list_preserves_submission_order() {
    let camera = Camera::default();
    let first = Mesh::new(
        Geometry::from_positions(vec![[0.0, 0.0, 0.0]]),
        BasicMaterial::default(),
    );
    let second = Mesh::new(
        Geometry::from_positions(vec![[1.0, 0.0, 0.0]]),
        BasicMaterial::default(),
    );
    let mut list = DrawList::new(&camera);
    list.push(&first);
    list.push(&second);

    assert_eq!(
        list.iter()
            .map(|item| item.mesh().geometry().positions()[0])
            .collect::<Vec<_>>(),
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    );
}
