use drackpad::homography::{
    Homography, HomographyError, Orientation, Point2d, RectSize, a4_size, rectify_matrix,
    warp_to_rect,
};
use drackpad::smoothing::Point2;
use opencv::core::{CV_8UC3, Mat, Point, Scalar};
use opencv::imgproc;
use opencv::prelude::*;

const EPS: f64 = 1e-9;

fn p(x: f64, y: f64) -> Point2d {
    Point2d::new(x, y)
}

fn trapezoid() -> [Point2d; 4] {
    [p(30.0, 10.0), p(70.0, 10.0), p(100.0, 50.0), p(0.0, 50.0)]
}

fn rect_100x50() -> [Point2d; 4] {
    [p(0.0, 0.0), p(100.0, 0.0), p(100.0, 50.0), p(0.0, 50.0)]
}

const EXPECTED: [[f64; 3]; 3] = [[4.0, 3.0, -150.0], [0.0, 5.0, -50.0], [0.0, 0.06, 1.0]];

fn assert_matrix(actual: [[f64; 3]; 3], expected: [[f64; 3]; 3]) {
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (actual[r][c] - expected[r][c]).abs() < EPS,
                "m[{r}][{c}]: {} vs {}",
                actual[r][c],
                expected[r][c]
            );
        }
    }
}

fn assert_point(actual: Point2d, x: f64, y: f64) {
    assert!(
        (actual.x - x).abs() < EPS && (actual.y - y).abs() < EPS,
        "{actual:?} vs ({x}, {y})"
    );
}

#[test]
fn t1_trapezoid_to_rect_matrix() {
    let h = Homography::from_points(trapezoid(), rect_100x50()).unwrap();
    assert_matrix(h.matrix(), EXPECTED);
}

#[test]
fn t2_identity() {
    let h = Homography::from_points(rect_100x50(), rect_100x50()).unwrap();
    assert_matrix(
        h.matrix(),
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    );
}

#[test]
fn t3_map_point() {
    let h = Homography::from_points(trapezoid(), rect_100x50()).unwrap();
    let dst = rect_100x50();
    for (s, d) in trapezoid().iter().zip(dst.iter()) {
        assert_point(h.map_point(*s).unwrap(), d.x, d.y);
    }
    assert_point(h.map_point(p(50.0, 30.0)).unwrap(), 50.0, 250.0 / 7.0);
}

#[test]
fn t4_rectify_matrix_uses_w_minus_1() {
    let size = RectSize {
        width: 101,
        height: 51,
    };
    let h = rectify_matrix(trapezoid(), size).unwrap();
    assert_matrix(h.matrix(), EXPECTED);
    assert_point(h.map_point(trapezoid()[2]).unwrap(), 100.0, 50.0);
}

#[test]
fn t5_errors() {
    let big = RectSize {
        width: 101,
        height: 51,
    };
    let rm = |c: [Point2d; 4]| rectify_matrix(c, big).unwrap_err();

    assert_eq!(
        rm([p(0.0, 0.0), p(0.0, 0.0), p(100.0, 50.0), p(0.0, 50.0)]),
        HomographyError::DuplicateCorners { a: 0, b: 1 }
    );
    assert_eq!(
        rm([p(0.0, 0.0), p(0.5, 0.0), p(100.0, 50.0), p(0.0, 50.0)]),
        HomographyError::DuplicateCorners { a: 0, b: 1 }
    );
    // exactly 1.0 px apart is not a duplicate
    assert!(
        rectify_matrix(
            [p(0.0, 0.0), p(1.0, 0.0), p(100.0, 50.0), p(0.0, 50.0)],
            big
        )
        .is_ok()
    );
    assert_eq!(
        rm([p(0.0, 0.0), p(50.0, 0.0), p(100.0, 0.0), p(0.0, 50.0)]),
        HomographyError::Collinear { indices: [0, 1, 2] }
    );
    assert!(matches!(
        rm([p(0.0, 0.0), p(30.0, 0.0), p(60.0, 0.0), p(90.0, 0.0)]),
        HomographyError::Collinear { .. }
    ));
    // twisted: TL, TR, BL, BR
    assert_eq!(
        rm([p(0.0, 0.0), p(100.0, 0.0), p(0.0, 50.0), p(100.0, 50.0)]),
        HomographyError::NotConvex
    );
    // concave
    assert_eq!(
        rm([p(0.0, 0.0), p(100.0, 0.0), p(30.0, 20.0), p(0.0, 50.0)]),
        HomographyError::NotConvex
    );
    // counter-clockwise: TL, BL, BR, TR
    assert_eq!(
        rm([p(0.0, 0.0), p(0.0, 50.0), p(100.0, 50.0), p(100.0, 0.0)]),
        HomographyError::ReversedOrder
    );
    assert_eq!(
        rm([p(0.0, 0.0), p(100.0, 0.0), p(f64::NAN, 50.0), p(0.0, 50.0)]),
        HomographyError::NonFiniteCorner { index: 2 }
    );
    assert_eq!(
        rectify_matrix(
            trapezoid(),
            RectSize {
                width: 1,
                height: 51
            }
        )
        .unwrap_err(),
        HomographyError::InvalidSize {
            width: 1,
            height: 51
        }
    );
    assert_eq!(
        rectify_matrix(
            trapezoid(),
            RectSize {
                width: 101,
                height: 1
            }
        )
        .unwrap_err(),
        HomographyError::InvalidSize {
            width: 101,
            height: 1
        }
    );

    let h = Homography::from_points(trapezoid(), rect_100x50()).unwrap();
    assert_eq!(
        h.map_point(p(0.0, -50.0 / 3.0)).unwrap_err(),
        HomographyError::PointAtInfinity
    );
    assert_eq!(
        h.map_point(p(f64::NAN, 1.0)).unwrap_err(),
        HomographyError::NonFinitePoint
    );
    let msg = HomographyError::Collinear { indices: [0, 1, 2] }.to_string();
    assert!(msg.contains("collinear"), "{msg}");
}

#[test]
fn t6_a4_size() {
    let s = |w, h| RectSize {
        width: w,
        height: h,
    };
    assert_eq!(a4_size(297, Orientation::Landscape).unwrap(), s(297, 210));
    assert_eq!(a4_size(1000, Orientation::Landscape).unwrap(), s(1000, 707));
    assert_eq!(a4_size(1000, Orientation::Portrait).unwrap(), s(707, 1000));
    assert_eq!(
        a4_size(2970, Orientation::Landscape).unwrap(),
        s(2970, 2100)
    );
    assert_eq!(a4_size(3, Orientation::Landscape).unwrap(), s(3, 2));
    assert_eq!(
        a4_size(2, Orientation::Landscape).unwrap_err(),
        HomographyError::InvalidSize {
            width: 2,
            height: 1
        }
    );
    for long in 3u32..=5000 {
        let float = (long as f64 * 210.0 / 297.0).round() as u64;
        let int = (long as u64 * 420 + 297) / 594;
        assert_eq!(float, int, "long={long}");
        assert_eq!(
            a4_size(long, Orientation::Landscape).unwrap().height as u64,
            int
        );
    }
}

fn black(w: i32, h: i32) -> Mat {
    Mat::new_rows_cols_with_default(h, w, CV_8UC3, Scalar::all(0.0)).unwrap()
}

fn scaled_trapezoid(k: f64) -> [Point2d; 4] {
    trapezoid().map(|q| p(q.x * k, q.y * k))
}

#[test]
fn t7_map_point_matches_warp() {
    let corners = scaled_trapezoid(4.0);
    let spots = [
        (40.0, 20.0),
        (50.0, 30.0),
        (80.0, 25.0),
        (60.0, 14.0),
        (90.0, 45.0),
    ];
    let mut img = black(400, 300);
    for (x, y) in spots {
        let (sx, sy) = ((x * 4.0) as i32, (y * 4.0) as i32);
        for dy in -1..=1 {
            for dx in -1..=1 {
                *img.at_2d_mut::<opencv::core::Vec3b>(sy + dy, sx + dx)
                    .unwrap() = opencv::core::Vec3b::from([255, 255, 255]);
            }
        }
    }
    let size = RectSize {
        width: 201,
        height: 101,
    };
    let h = rectify_matrix(corners, size).unwrap();
    let out = warp_to_rect(&img, &h, size).unwrap();
    for (x, y) in spots {
        let m = h.map_point(p(x * 4.0, y * 4.0)).unwrap();
        let (cx, cy) = (m.x.round() as i32, m.y.round() as i32);
        let mut best = (0u8, 0i32, 0i32);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let v = out.at_2d::<opencv::core::Vec3b>(cy + dy, cx + dx).unwrap()[0];
                if v > best.0 {
                    best = (v, cx + dx, cy + dy);
                }
            }
        }
        assert!(best.0 > 0, "no spot near {m:?}");
        assert!(
            (best.1 as f64 - m.x).abs() <= 1.5 && (best.2 as f64 - m.y).abs() <= 1.5,
            "peak {best:?} vs {m:?}"
        );
    }
}

#[test]
fn t8_synthetic_paper_is_rectified() {
    let corners = scaled_trapezoid(4.0);
    let mut img = black(400, 300);
    let pts: opencv::core::Vector<Point> = corners
        .iter()
        .map(|q| Point::new(q.x as i32, q.y as i32))
        .collect();
    imgproc::fill_convex_poly(&mut img, &pts, Scalar::all(255.0), imgproc::LINE_8, 0).unwrap();
    let size = RectSize {
        width: 200,
        height: 100,
    };
    let h = rectify_matrix(corners, size).unwrap();
    let out = warp_to_rect(&img, &h, size).unwrap();
    assert_eq!((out.cols(), out.rows()), (200, 100));
    assert_eq!(out.typ(), CV_8UC3);
    for y in 3..97 {
        for x in 3..197 {
            let v = out.at_2d::<opencv::core::Vec3b>(y, x).unwrap();
            assert!(
                v[0] >= 250 && v[1] >= 250 && v[2] >= 250,
                "({x},{y}) = {v:?}"
            );
        }
    }
}

#[test]
fn t9_from_smoothing_point() {
    let q = Point2d::from(Point2 { x: 1.5, y: -2.0 });
    assert_eq!(q, p(1.5, -2.0));
}

#[test]
fn t10_min_sin_boundary() {
    // At corner 1 the edges are (1000,0) and (1000,d): sin ~= d/1000.
    let quad = |d: f64| [p(0.0, 0.0), p(1000.0, 0.0), p(2000.0, d), p(0.0, 500.0)];
    // sin ~ 5e-7 < 1e-6: collinear
    assert_eq!(
        drackpad::homography::validate_quad(&quad(5e-4)),
        Err(HomographyError::Collinear { indices: [0, 1, 2] })
    );
    // sin ~ 2e-6 > 1e-6: accepted
    assert_eq!(drackpad::homography::validate_quad(&quad(2e-3)), Ok(()));
}

#[test]
fn t11_from_points_validates_dst() {
    let src = rect_100x50();
    let err = |dst| Homography::from_points(src, dst).unwrap_err();
    assert_eq!(
        err([p(0.0, 0.0), p(0.0, 0.0), p(100.0, 50.0), p(0.0, 50.0)]),
        HomographyError::DuplicateCorners { a: 0, b: 1 }
    );
    assert_eq!(
        err([p(0.0, 0.0), p(50.0, 0.0), p(100.0, 0.0), p(0.0, 50.0)]),
        HomographyError::Collinear { indices: [0, 1, 2] }
    );
    assert_eq!(
        err([p(0.0, 0.0), p(100.0, 0.0), p(30.0, 20.0), p(0.0, 50.0)]),
        HomographyError::NotConvex
    );
    assert_eq!(
        err([p(0.0, 0.0), p(0.0, 50.0), p(100.0, 50.0), p(100.0, 0.0)]),
        HomographyError::ReversedOrder
    );
}

#[test]
fn t12_size_must_fit_i32() {
    let too_big = |width, height| HomographyError::SizeTooLarge { width, height };
    for (w, h) in [
        (4294967295u32, 100u32),
        (2147483648, 100),
        (100, 4294967295),
    ] {
        let size = RectSize {
            width: w,
            height: h,
        };
        assert_eq!(
            rectify_matrix(trapezoid(), size).unwrap_err(),
            too_big(w, h)
        );
        // warp_to_rect rejects before allocating anything
        let img = black(10, 10);
        let h3 = Homography::from_points(trapezoid(), rect_100x50()).unwrap();
        let e = warp_to_rect(&img, &h3, size).unwrap_err();
        assert_eq!(e.downcast_ref::<HomographyError>(), Some(&too_big(w, h)));
    }
    assert_eq!(
        a4_size(4294967295, Orientation::Landscape).unwrap_err(),
        too_big(4294967295, 3036845562)
    );
    // i32::MAX itself is a valid conversion (no allocation: matrix only)
    let ok = RectSize {
        width: 2147483647,
        height: 100,
    };
    assert!(rectify_matrix(trapezoid(), ok).is_ok());
    assert!(
        HomographyError::SizeTooLarge {
            width: 1,
            height: 1
        }
        .to_string()
        .contains("too large")
    );
}

fn square(base: f64, side: f64) -> [Point2d; 4] {
    [
        p(base, base),
        p(base + side, base),
        p(base + side, base + side),
        p(base, base + side),
    ]
}

const SIZE_100: RectSize = RectSize {
    width: 100,
    height: 100,
};

#[test]
fn t13_corners_collapsing_in_f32_are_rejected() {
    // Distinct in f64, but all within one f32 step (spacing 8 at 1e8).
    let corners = [
        p(100000000.0, 100000000.0),
        p(100000001.0, 100000000.0),
        p(100000001.0, 100000001.0),
        p(100000000.0, 100000001.5),
    ];
    assert!(drackpad::homography::validate_quad(&corners).is_ok());
    assert_eq!(
        rectify_matrix(corners, SIZE_100).unwrap_err(),
        HomographyError::DuplicateCorners { a: 0, b: 1 }
    );
    // dst is checked after rounding too
    assert_eq!(
        Homography::from_points(rect_100x50(), corners).unwrap_err(),
        HomographyError::DuplicateCorners { a: 0, b: 1 }
    );
    assert_eq!(
        rectify_matrix(square(1e9, 8.0), SIZE_100).unwrap_err(),
        HomographyError::DuplicateCorners { a: 0, b: 1 }
    );
}

#[test]
fn t14_zero_matrix_from_opencv_is_singular() {
    // Survives validation in f64 and after rounding to f32 (one f32 step
    // apart), yet OpenCV returns the zero matrix [0 0 0; 0 0 0; 0 0 1].
    assert_eq!(
        rectify_matrix(square(1e15, 7e7), SIZE_100).unwrap_err(),
        HomographyError::Singular
    );
}

#[test]
fn t15_large_but_valid_inputs_still_work() {
    for (base, side) in [(1.6e7, 100.0), (1e6, 64.0), (0.0, 100.0)] {
        let h = rectify_matrix(square(base, side), SIZE_100).unwrap();
        let m = h.map_point(p(base + side, base + side)).unwrap();
        assert!(
            (m.x - 99.0).abs() < 1.0 && (m.y - 99.0).abs() < 1.0,
            "{m:?}"
        );
    }
}
