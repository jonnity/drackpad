//! Smoothing of 2D coordinate time series.

use std::fmt;

/// A 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2 {
    pub x: f32,
    pub y: f32,
}

/// A stateful filter over a stream of points (constant frame interval assumed).
pub trait Smoother {
    /// Feeds one frame's detected point and returns the smoothed point.
    fn update(&mut self, p: Point2) -> Point2;
    /// Discards state; the next input becomes the new initial value.
    fn reset(&mut self);
}

/// Default smoothing coefficient.
pub const DEFAULT_ALPHA: f32 = 0.5;

/// Returned when the coefficient is not in `(0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InvalidAlpha(pub f32);

impl fmt::Display for InvalidAlpha {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alpha must be in (0, 1], got {}", self.0)
    }
}

impl std::error::Error for InvalidAlpha {}

/// Exponential moving average: `out = alpha * p + (1 - alpha) * prev_out`.
/// The first input (or the first after `reset`) is returned unchanged.
#[derive(Debug, Clone)]
pub struct Ema {
    alpha: f32,
    state: Option<Point2>,
}

impl Ema {
    /// Creates a filter. `alpha` must satisfy `0 < alpha <= 1`; 1 means no smoothing.
    pub fn new(alpha: f32) -> Result<Self, InvalidAlpha> {
        if alpha > 0.0 && alpha <= 1.0 {
            Ok(Self { alpha, state: None })
        } else {
            Err(InvalidAlpha(alpha))
        }
    }

    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

impl Default for Ema {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            state: None,
        }
    }
}

impl Smoother for Ema {
    fn update(&mut self, p: Point2) -> Point2 {
        let a = self.alpha;
        let out = match self.state {
            None => p,
            Some(s) => Point2 {
                x: a * p.x + (1.0 - a) * s.x,
                y: a * p.y + (1.0 - a) * s.y,
            },
        };
        self.state = Some(out);
        out
    }

    fn reset(&mut self) {
        self.state = None;
    }
}

/// Smooths a series; a missing frame (`None`) resets the filter and yields `None`.
pub fn smooth_series<S: Smoother + ?Sized>(
    s: &mut S,
    input: &[Option<Point2>],
) -> Vec<Option<Point2>> {
    input
        .iter()
        .map(|p| match p {
            Some(p) => Some(s.update(*p)),
            None => {
                s.reset();
                None
            }
        })
        .collect()
}
