//! Tests CPU clip validation and paired reservation rollback.

use super::*;

fn identity() -> [[f32; 4]; 4] {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

fn check(position: [f32; 3], matrix: [[f32; 4]; 4]) -> Result<(), DrawStartError> {
    validate_clip(&[position], &[0, 0, 0], &matrix)
}

#[test]
fn clip_gate_accepts_closed_clip_volume() {
    assert_eq!(check([1.0, -1.0, 1.0], identity()), Ok(()));
}

#[test]
fn clip_gate_rejects_nonfinite_position_and_nonfinite_math() {
    assert_eq!(
        check([f32::NAN, 0.0, 0.0], identity()),
        Err(DrawStartError::NonFinitePosition)
    );
    let mut product_overflow = identity();
    product_overflow[0][0] = f32::MAX;
    assert_eq!(
        check([f32::MAX, 0.0, 0.0], product_overflow),
        Err(DrawStartError::NonFiniteClipPosition)
    );
    let mut accumulation_overflow = identity();
    accumulation_overflow[0][0] = 1.0;
    accumulation_overflow[1][0] = 1.0;
    assert_eq!(
        check([f32::MAX, f32::MAX, 0.0], accumulation_overflow),
        Err(DrawStartError::NonFiniteClipPosition)
    );
}

#[test]
fn clip_gate_rejects_w_and_each_axis_boundary() {
    let mut no_w = identity();
    no_w[3][3] = 0.0;
    assert_eq!(
        check([0.0, 0.0, 0.0], no_w),
        Err(DrawStartError::ClipWNonPositive)
    );
    assert_eq!(
        check([1.1, 0.0, 0.0], identity()),
        Err(DrawStartError::ClipOutOfBounds)
    );
    assert_eq!(
        check([0.0, -1.1, 0.0], identity()),
        Err(DrawStartError::ClipOutOfBounds)
    );
    assert_eq!(
        check([0.0, 0.0, -0.1], identity()),
        Err(DrawStartError::ClipOutOfBounds)
    );
    assert_eq!(
        check([0.0, 0.0, 1.1], identity()),
        Err(DrawStartError::ClipOutOfBounds)
    );
}

#[test]
fn paired_gate_second_failure_rolls_back_the_first_gate_atomically() {
    use core::cell::Cell;
    let first_released = Cell::new(false);
    let result: Result<(u8, u8), PairReservationError<&'static str>> =
        reserve_pair(Ok(7), || Err("texture busy"), |_| first_released.set(true));
    assert!(matches!(
        result,
        Err(PairReservationError::Second("texture busy"))
    ));
    assert!(first_released.get());
    let first_error: Result<(u8, u8), PairReservationError<&'static str>> = reserve_pair(
        Err("mesh busy"),
        || Ok(9),
        |_| unreachable!("first failure cannot roll back"),
    );
    assert!(matches!(
        first_error,
        Err(PairReservationError::First("mesh busy"))
    ));
}
