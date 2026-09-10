//! Tests the fixed uniform ABI and fail-closed numeric validation.

use super::*;

#[test]
fn serializes_column_major_projection_times_view_and_color() {
    let view = [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ];
    let projection = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 3.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 0.0],
        [7.0, 0.0, 0.0, 11.0],
    ];
    let uniform = FrameUniform::new(
        &Camera::new(view, projection),
        &BasicMaterial::new([0.0, 0.5, 1.0, 0.25]),
    )
    .unwrap();
    let words: Vec<_> = uniform
        .bytes()
        .chunks_exact(4)
        .map(|word| f32::from_bits(u32::from_le_bytes(word.try_into().unwrap())))
        .collect();
    assert_eq!(
        words,
        vec![
            30.0, 6.0, 15.0, 44.0, 66.0, 18.0, 35.0, 88.0, 102.0, 30.0, 55.0, 132.0, 138.0, 42.0,
            75.0, 176.0, 0.0, 0.5, 1.0, 0.25,
        ]
    );
}

#[test]
fn rejects_non_finite_out_of_range_and_overflowing_inputs() {
    assert_eq!(
        FrameUniform::new(
            &Camera::new([[f32::NAN; 4]; 4], [[0.0; 4]; 4]),
            &BasicMaterial::default(),
        ),
        Err(FrameUniformError::NonFiniteCamera)
    );
    assert_eq!(
        FrameUniform::new(
            &Camera::default(),
            &BasicMaterial::new([0.0, 0.0, 0.0, f32::INFINITY])
        ),
        Err(FrameUniformError::NonFiniteColor)
    );
    assert_eq!(
        FrameUniform::new(
            &Camera::default(),
            &BasicMaterial::new([1.1, 0.0, 0.0, 1.0])
        ),
        Err(FrameUniformError::ColorOutOfRange)
    );
    assert_eq!(
        FrameUniform::new(
            &Camera::new([[f32::MAX; 4]; 4], [[f32::MAX; 4]; 4]),
            &BasicMaterial::default(),
        ),
        Err(FrameUniformError::MatrixProductNonFinite)
    );
}
