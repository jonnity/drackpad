//! CSV input/output for comparing raw and smoothed coordinate traces.

use crate::smoothing::{Point2, Smoother, smooth_series};
use std::fmt;

const IN_HEADER: &str = "frame,x,y";
const OUT_HEADER: &str = "frame,raw_x,raw_y,smooth_x,smooth_y";

/// One input row. `point` is `None` for a missing detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    pub frame: u64,
    pub point: Option<Point2>,
}

/// A CSV problem, with a 1-based line number.
#[derive(Debug, Clone, PartialEq)]
pub struct CsvError {
    pub line: usize,
    pub message: String,
}

impl fmt::Display for CsvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for CsvError {}

fn err(line: usize, message: impl Into<String>) -> CsvError {
    CsvError {
        line,
        message: message.into(),
    }
}

fn parse_coord(s: &str, name: &str, line: usize) -> Result<f32, CsvError> {
    let v: f32 = s
        .trim()
        .parse()
        .map_err(|_| err(line, format!("invalid {name}: {s:?}")))?;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(err(line, format!("{name} is not finite: {s:?}")))
    }
}

/// Parses `frame,x,y` CSV. Rows with both `x` and `y` empty are missing detections.
pub fn parse_input(text: &str) -> Result<Vec<Sample>, CsvError> {
    if text.is_empty() {
        return Err(err(1, "missing header (expected \"frame,x,y\")"));
    }
    let body = text.strip_suffix('\n').unwrap_or(text);
    let mut out = Vec::new();
    for (i, raw) in body.split('\n').enumerate() {
        let line = i + 1;
        let row = raw.strip_suffix('\r').unwrap_or(raw);
        if line == 1 {
            if row != IN_HEADER {
                return Err(err(1, format!("header must be {IN_HEADER:?}, got {row:?}")));
            }
            continue;
        }
        let cols: Vec<&str> = row.split(',').collect();
        if cols.len() != 3 {
            return Err(err(line, format!("expected 3 columns, got {}", cols.len())));
        }
        let frame: u64 = cols[0]
            .trim()
            .parse()
            .map_err(|_| err(line, format!("invalid frame: {:?}", cols[0])))?;
        let (xs, ys) = (cols[1].trim(), cols[2].trim());
        let point = match (xs.is_empty(), ys.is_empty()) {
            (true, true) => None,
            (false, false) => Some(Point2 {
                x: parse_coord(xs, "x", line)?,
                y: parse_coord(ys, "y", line)?,
            }),
            _ => return Err(err(line, "only one of x and y is empty")),
        };
        out.push(Sample { frame, point });
    }
    Ok(out)
}

/// Renders the comparison CSV. `smoothed` must be parallel to `samples`.
pub fn write_comparison(samples: &[Sample], smoothed: &[Option<Point2>]) -> String {
    let mut s = String::from(OUT_HEADER);
    s.push('\n');
    for (sample, sm) in samples.iter().zip(smoothed) {
        match (sample.point, sm) {
            (Some(r), Some(m)) => s.push_str(&format!(
                "{},{},{},{},{}\n",
                sample.frame, r.x, r.y, m.x, m.y
            )),
            _ => s.push_str(&format!("{},,,,\n", sample.frame)),
        }
    }
    s
}

/// Parses, smooths and renders in one step.
pub fn compare_csv(text: &str, smoother: &mut dyn Smoother) -> Result<String, CsvError> {
    let samples = parse_input(text)?;
    let input: Vec<Option<Point2>> = samples.iter().map(|s| s.point).collect();
    let smoothed = smooth_series(smoother, &input);
    Ok(write_comparison(&samples, &smoothed))
}
