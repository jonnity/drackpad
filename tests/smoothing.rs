use drackpad::smoothing::*;

fn p(x: f32, y: f32) -> Point2 {
    Point2 { x, y }
}
fn run(alpha: f32, xs: &[f32]) -> Vec<f32> {
    let mut e = Ema::new(alpha).unwrap();
    xs.iter().map(|&x| e.update(p(x, 0.0)).x).collect()
}

#[test]
fn first_frame_and_reset_pass_through() {
    let mut e = Ema::default();
    assert_eq!(e.update(p(3.0, 4.0)), p(3.0, 4.0));
    e.update(p(100.0, 100.0));
    e.reset();
    assert_eq!(e.update(p(7.0, 8.0)), p(7.0, 8.0));
}

#[test]
fn constant_input_stays_constant() {
    let mut e = Ema::default();
    for _ in 0..10 {
        assert_eq!(e.update(p(100.0, 200.0)), p(100.0, 200.0));
    }
}

#[test]
fn converges_to_new_level() {
    let mut xs = vec![0.0];
    xs.extend([100.0; 20]);
    let out = run(0.5, &xs);
    let mut prev = f32::MAX;
    for o in &out[1..] {
        let d = (o - 100.0).abs();
        assert!(d < prev);
        prev = d;
    }
    assert!((out[20] - 100.0).abs() < 1e-3);
}

#[test]
fn outlier_effect_shrinks() {
    let xs: Vec<f32> = (0..16)
        .map(|i| if i == 3 { 200.0 } else { 100.0 })
        .collect();
    let out = run(0.5, &xs);
    assert_eq!(out[3] - 100.0, 50.0);
    for k in 0..=12 {
        assert_eq!(out[3 + k] - 100.0, 50.0 * 0.5f32.powi(k as i32));
    }
    assert_eq!(out[10] - 100.0, 0.390625);
    let mut e = Ema::default();
    for &x in &xs {
        assert_eq!(e.update(p(x, 100.0)).y, 100.0);
    }
}

#[test]
fn ramp_lag_is_one_frame() {
    let xs: Vec<f32> = (0..21).map(|k| 10.0 * k as f32).collect();
    let out = run(0.5, &xs);
    for k in 0..21 {
        assert!(xs[k] - out[k] <= 10.0 + 1e-3);
    }
    assert!((xs[20] - out[20] - 10.0).abs() < 1e-3);
}

#[test]
fn step_response() {
    let xs = [
        0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];
    let out = run(0.5, &xs);
    assert_eq!(
        out,
        vec![
            0.0, 0.0, 50.0, 75.0, 87.5, 93.75, 96.875, 98.4375, 99.21875, 99.609375
        ]
    );
    assert!(out[5] < 95.0 && out[6] >= 95.0);
}

#[test]
fn x_and_y_independent() {
    let mut e = Ema::default();
    for k in 0..10 {
        assert_eq!(e.update(p(10.0 * k as f32, 5.0)).y, 5.0);
    }
}

#[test]
fn missing_frame_resets() {
    let mut e = Ema::default();
    let input = [
        Some(p(0.0, 0.0)),
        Some(p(0.0, 0.0)),
        None,
        Some(p(100.0, 100.0)),
    ];
    assert_eq!(smooth_series(&mut e, &input), input.to_vec());
}

#[test]
fn default_alpha() {
    assert_eq!(DEFAULT_ALPHA, 0.5);
    assert_eq!(Ema::default().alpha(), 0.5);
}

#[test]
fn alpha_range() {
    for a in [
        0.0,
        -0.1,
        f32::NAN,
        1.0001,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert!(Ema::new(a).is_err(), "{a}");
    }
    for a in [1.0, f32::MIN_POSITIVE, 0.5] {
        assert!(Ema::new(a).is_ok(), "{a}");
    }
}

#[test]
fn alpha_one_is_identity() {
    let xs = [1.5, 200.0, 3.25, -7.0, 100.0, 0.1];
    assert_eq!(run(1.0, &xs), xs.to_vec());
}

#[test]
fn alpha_changes_speed() {
    let xs = [0.0, 100.0, 100.0];
    assert_eq!(run(0.25, &xs)[1], 25.0);
    assert_eq!(run(0.75, &xs)[1], 75.0);
}

#[test]
fn usable_as_trait_object() {
    let mut e = Ema::default();
    let s: &mut dyn Smoother = &mut e;
    let out = smooth_series(s, &[Some(p(1.0, 1.0))]);
    assert_eq!(out, vec![Some(p(1.0, 1.0))]);
}

fn run_xy(alpha: f32, pts: &[(f32, f32)]) -> Vec<(f32, f32)> {
    let mut e = Ema::new(alpha).unwrap();
    pts.iter()
        .map(|&(x, y)| {
            let o = e.update(p(x, y));
            (o.x, o.y)
        })
        .collect()
}

#[test]
fn y_step_response_matches_x_table() {
    let ys = [
        0.0, 0.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
    ];
    let pts: Vec<(f32, f32)> = ys.iter().map(|&y| (7.0, y)).collect();
    let out = run_xy(0.5, &pts);
    let expected = [
        0.0, 0.0, 50.0, 75.0, 87.5, 93.75, 96.875, 98.4375, 99.21875, 99.609375,
    ];
    for (o, e) in out.iter().zip(expected) {
        assert_eq!(o.0, 7.0);
        assert_eq!(o.1, e);
    }
}

#[test]
fn y_ramp_lags() {
    let pts: Vec<(f32, f32)> = (0..4).map(|k| (3.0, 10.0 * k as f32)).collect();
    let ys: Vec<f32> = run_xy(0.5, &pts).iter().map(|o| o.1).collect();
    // 0, 0.5*10+0.5*0, 0.5*20+0.5*5, 0.5*30+0.5*12.5
    assert_eq!(ys, vec![0.0, 5.0, 12.5, 21.25]);
}

#[test]
fn x_and_y_move_simultaneously_with_different_values() {
    let pts = [(0.0, 200.0), (100.0, 0.0), (100.0, 0.0), (100.0, 0.0)];
    let out = run_xy(0.5, &pts);
    assert_eq!(
        out,
        vec![(0.0, 200.0), (50.0, 100.0), (75.0, 50.0), (87.5, 25.0)]
    );
}

#[test]
fn alpha_quarter_weights_both_axes() {
    // x: 0 -> 100 (step up), y: 100 -> 0 (step down). Hand computed:
    // x: 25, 0.25*100+0.75*25 = 43.75, 0.25*100+0.75*43.75 = 57.8125
    // y: 0.75*100 = 75, 0.75*75 = 56.25, 0.75*56.25 = 42.1875
    let pts = [(0.0, 100.0), (100.0, 0.0), (100.0, 0.0), (100.0, 0.0)];
    let out = run_xy(0.25, &pts);
    assert_eq!(
        out,
        vec![
            (0.0, 100.0),
            (25.0, 75.0),
            (43.75, 56.25),
            (57.8125, 42.1875)
        ]
    );
}

#[test]
fn alpha_three_quarters_weights_both_axes() {
    // x: 0.75*100 = 75, 0.75*100+0.25*75 = 93.75
    // y: 0.25*100 = 25, 0.25*25 = 6.25
    let pts = [(0.0, 100.0), (100.0, 0.0), (100.0, 0.0)];
    let out = run_xy(0.75, &pts);
    assert_eq!(out, vec![(0.0, 100.0), (75.0, 25.0), (93.75, 6.25)]);
}
