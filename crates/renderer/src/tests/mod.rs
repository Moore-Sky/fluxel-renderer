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
fn basic_material_rejects_each_invalid_component_and_keeps_unit_boundaries() {
    for component in 0..4 {
        let mut color = [0.0; 4];
        color[component] = f32::NAN;
        assert_eq!(
            BasicMaterial::new(color),
            Err(BasicMaterialError::NonFinite { component })
        );

        color[component] = -f32::MIN_POSITIVE;
        assert_eq!(
            BasicMaterial::new(color),
            Err(BasicMaterialError::OutOfRange { component })
        );
        color[component] = 1.0 + f32::EPSILON;
        assert_eq!(
            BasicMaterial::new(color),
            Err(BasicMaterialError::OutOfRange { component })
        );
    }
    assert!(BasicMaterial::new([0.0, 1.0, 0.0, 1.0]).is_ok());
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

#[test]
fn model_transform_validates_affinity_and_draw_list_preserves_placements() {
    let camera = Camera::default();
    let mesh = Mesh::new(
        Geometry::from_positions(vec![[0.0, 0.0, 0.0]]),
        BasicMaterial::default(),
    );
    let translation = ModelTransform::from_column_major([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [2.0, 3.0, 4.0, 1.0],
    ])
    .unwrap();
    assert_eq!(ModelTransform::default(), ModelTransform::IDENTITY);
    assert_eq!(translation.world_from_model()[3], [2.0, 3.0, 4.0, 1.0]);
    assert_eq!(
        ModelTransform::from_column_major([[f32::NAN; 4]; 4]),
        Err(ModelTransformError::NonFinite)
    );
    assert_eq!(
        ModelTransform::from_column_major([
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        Err(ModelTransformError::NonAffine)
    );

    let mut list = DrawList::new(&camera);
    list.push(&mesh);
    list.push_transformed(&mesh, translation);
    assert_eq!(list.len(), 2);
    assert_eq!(
        list.iter().next().unwrap().transform(),
        ModelTransform::IDENTITY
    );
    assert_eq!(list.iter().nth(1).unwrap().transform(), translation);
}
