//! Perspective transform for rectifying a sheet of paper seen at an angle.
//!
//! Corners are image pixel coordinates ordered top-left, top-right,
//! bottom-right, bottom-left. The matrix comes from OpenCV's
//! `getPerspectiveTransform`; corner validation and point mapping are pure Rust.

use opencv::core::{CV_64F, Mat, Point2f, Scalar, Size};
use opencv::imgproc;
use opencv::prelude::*;

/// Two corners closer than this (in pixels) count as duplicates.
pub const MIN_CORNER_DISTANCE: f64 = 1.0;
/// Three consecutive corners whose edges have a sine of the angle at or below
/// this value count as collinear.
pub const MIN_SIN: f64 = 1e-6;
/// Smallest width/height of the output rectangle (`W-1` must be at least 1).
pub const MIN_RECT_SIDE: u32 = 2;

const EPS_W: f64 = 1e-12;

/// A point in pixel coordinates (f64).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2d {
    pub x: f64,
    pub y: f64,
}

impl Point2d {
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

impl From<crate::smoothing::Point2> for Point2d {
    fn from(p: crate::smoothing::Point2) -> Self {
        Self::new(p.x as f64, p.y as f64)
    }
}

/// Output rectangle size in pixels. No default (does not implement `Default`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RectSize {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Orientation {
    Landscape,
    Portrait,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HomographyError {
    NonFiniteCorner { index: usize },
    DuplicateCorners { a: usize, b: usize },
    Collinear { indices: [usize; 3] },
    ReversedOrder,
    NotConvex,
    InvalidSize { width: u32, height: u32 },
    SizeTooLarge { width: u32, height: u32 },
    Singular,
    OpenCv(String),
    NonFinitePoint,
    PointAtInfinity,
}

impl std::fmt::Display for HomographyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteCorner { index } => write!(f, "corner {index} is not a finite number"),
            Self::DuplicateCorners { a, b } => write!(f, "corners {a} and {b} are duplicated"),
            Self::Collinear { indices: [a, b, c] } => {
                write!(f, "corners {a}, {b}, {c} are collinear")
            }
            Self::ReversedOrder => write!(
                f,
                "corners are in reversed order (expected top-left, top-right, bottom-right, bottom-left)"
            ),
            Self::NotConvex => write!(
                f,
                "corners do not form a convex quadrilateral in the expected order"
            ),
            Self::InvalidSize { width, height } => write!(
                f,
                "invalid size {width}x{height} (each side must be at least {MIN_RECT_SIDE})"
            ),
            Self::SizeTooLarge { width, height } => write!(
                f,
                "size {width}x{height} is too large (each side must be at most {})",
                i32::MAX
            ),
            Self::Singular => write!(f, "perspective transform is singular"),
            Self::OpenCv(e) => write!(f, "opencv error: {e}"),
            Self::NonFinitePoint => write!(f, "point is not a finite number"),
            Self::PointAtInfinity => write!(f, "point maps to infinity"),
        }
    }
}

impl std::error::Error for HomographyError {}

/// 3x3 perspective transform, normalized so that `m[2][2] == 1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homography {
    m: [[f64; 3]; 3],
}

impl RectSize {
    /// Checks the minimum side and that both sides fit in OpenCV's `i32`.
    fn to_cv_size(self) -> Result<Size, HomographyError> {
        let (width, height) = (self.width, self.height);
        if width < MIN_RECT_SIDE || height < MIN_RECT_SIDE {
            return Err(HomographyError::InvalidSize { width, height });
        }
        match (i32::try_from(width), i32::try_from(height)) {
            (Ok(w), Ok(h)) => Ok(Size::new(w, h)),
            _ => Err(HomographyError::SizeTooLarge { width, height }),
        }
    }
}

fn sub(a: Point2d, b: Point2d) -> (f64, f64) {
    (a.x - b.x, a.y - b.y)
}

/// Checks that `q` is a usable quadrilateral ordered top-left, top-right,
/// bottom-right, bottom-left (clockwise in image coordinates, y down).
pub fn validate_quad(q: &[Point2d; 4]) -> Result<(), HomographyError> {
    for (index, p) in q.iter().enumerate() {
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(HomographyError::NonFiniteCorner { index });
        }
    }
    for a in 0..4 {
        for b in (a + 1)..4 {
            let (dx, dy) = sub(q[a], q[b]);
            if dx.hypot(dy) < MIN_CORNER_DISTANCE {
                return Err(HomographyError::DuplicateCorners { a, b });
            }
        }
    }
    let mut crosses = [0.0f64; 4];
    for i in 0..4 {
        let (p, n) = ((i + 3) % 4, (i + 1) % 4);
        let a = sub(q[i], q[p]);
        let b = sub(q[n], q[i]);
        let cross = a.0 * b.1 - a.1 * b.0;
        if cross.abs() <= MIN_SIN * a.0.hypot(a.1) * b.0.hypot(b.1) {
            return Err(HomographyError::Collinear { indices: [p, i, n] });
        }
        crosses[i] = cross;
    }
    if crosses.iter().all(|&c| c > 0.0) {
        Ok(())
    } else if crosses.iter().all(|&c| c < 0.0) {
        Err(HomographyError::ReversedOrder)
    } else {
        Err(HomographyError::NotConvex)
    }
}

fn to_f32(p: Point2d) -> Point2f {
    Point2f::new(p.x as f32, p.y as f32)
}

fn from_f32(p: Point2f) -> Point2d {
    Point2d::new(p.x as f64, p.y as f64)
}

fn cv_err(e: opencv::Error) -> HomographyError {
    HomographyError::OpenCv(e.to_string())
}

impl Homography {
    /// Perspective transform mapping the `src` corners onto the `dst` corners.
    pub fn from_points(src: [Point2d; 4], dst: [Point2d; 4]) -> Result<Self, HomographyError> {
        validate_quad(&src)?;
        validate_quad(&dst)?;
        // OpenCV takes f32: re-validate what it will actually see, since
        // rounding can merge or flatten corners that were fine in f64.
        let (src32, dst32) = (src.map(to_f32), dst.map(to_f32));
        validate_quad(&src32.map(from_f32))?;
        validate_quad(&dst32.map(from_f32))?;
        let mat = imgproc::get_perspective_transform_slice_def(src32, dst32).map_err(cv_err)?;
        let mut m = [[0.0f64; 3]; 3];
        for (r, row) in m.iter_mut().enumerate() {
            for (c, v) in row.iter_mut().enumerate() {
                *v = *mat.at_2d::<f64>(r as i32, c as i32).map_err(cv_err)?;
            }
        }
        let norm = m[2][2];
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        if m.iter().flatten().any(|v| !v.is_finite())
            || norm.abs() < EPS_W
            || !det.is_finite()
            || det == 0.0
        {
            return Err(HomographyError::Singular);
        }
        for v in m.iter_mut().flatten() {
            *v /= norm;
        }
        m[2][2] = 1.0;
        Ok(Self { m })
    }

    pub fn matrix(&self) -> [[f64; 3]; 3] {
        self.m
    }

    /// Maps a point in the source image to the rectified image (pixels).
    /// Points outside the paper are returned as is (no clamping).
    pub fn map_point(&self, p: Point2d) -> Result<Point2d, HomographyError> {
        if !p.x.is_finite() || !p.y.is_finite() {
            return Err(HomographyError::NonFinitePoint);
        }
        let m = &self.m;
        let w = m[2][0] * p.x + m[2][1] * p.y + m[2][2];
        if w.abs() < EPS_W {
            return Err(HomographyError::PointAtInfinity);
        }
        Ok(Point2d::new(
            (m[0][0] * p.x + m[0][1] * p.y + m[0][2]) / w,
            (m[1][0] * p.x + m[1][1] * p.y + m[1][2]) / w,
        ))
    }

    fn to_mat(self) -> opencv::Result<Mat> {
        let mut mat = Mat::new_rows_cols_with_default(3, 3, CV_64F, Scalar::all(0.0))?;
        for (r, row) in self.m.iter().enumerate() {
            for (c, v) in row.iter().enumerate() {
                *mat.at_2d_mut::<f64>(r as i32, c as i32)? = *v;
            }
        }
        Ok(mat)
    }
}

/// Matrix that maps `corners` (top-left, top-right, bottom-right, bottom-left)
/// onto `(0,0), (W-1,0), (W-1,H-1), (0,H-1)` of a `size` image.
pub fn rectify_matrix(
    corners: [Point2d; 4],
    size: RectSize,
) -> Result<Homography, HomographyError> {
    size.to_cv_size()?;
    let (w, h) = ((size.width - 1) as f64, (size.height - 1) as f64);
    let dst = [
        Point2d::new(0.0, 0.0),
        Point2d::new(w, 0.0),
        Point2d::new(w, h),
        Point2d::new(0.0, h),
    ];
    Homography::from_points(corners, dst)
}

/// A4 size from its long side. The short side is `long * 210 / 297` rounded
/// to the nearest integer; callers must not compute the rounding themselves.
pub fn a4_size(long_side_px: u32, orientation: Orientation) -> Result<RectSize, HomographyError> {
    let short = (long_side_px as f64 * 210.0 / 297.0).round() as u32;
    let (width, height) = match orientation {
        Orientation::Landscape => (long_side_px, short),
        Orientation::Portrait => (short, long_side_px),
    };
    let size = RectSize { width, height };
    size.to_cv_size()?;
    Ok(size)
}

/// Warps `src` to a `size` image with bilinear interpolation (black outside).
pub fn warp_to_rect(src: &Mat, h: &Homography, size: RectSize) -> anyhow::Result<Mat> {
    if src.empty() {
        anyhow::bail!("source image is empty");
    }
    let cv_size = size.to_cv_size()?;
    let m = h.to_mat()?;
    let mut dst = Mat::default();
    imgproc::warp_perspective(
        src,
        &mut dst,
        &m,
        cv_size,
        imgproc::INTER_LINEAR,
        opencv::core::BORDER_CONSTANT,
        Scalar::default(),
    )?;
    Ok(dst)
}
