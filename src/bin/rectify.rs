//! Usage: rectify <INPUT> <OUTPUT> --size <W>x<H> --corners <x1,y1,x2,y2,x3,y3,x4,y4>
//!
//! Warps the quadrilateral given by the four corners (top-left, top-right,
//! bottom-right, bottom-left, in source pixels) to a `W`x`H` image.

use drackpad::homography::{Point2d, RectSize, rectify_matrix, warp_to_rect};
use opencv::core::Vector;
use opencv::{imgcodecs, prelude::*};
use std::process::ExitCode;

const USAGE: &str =
    "usage: rectify <INPUT> <OUTPUT> --size <W>x<H> --corners <x1,y1,x2,y2,x3,y3,x4,y4>";

/// True if both paths currently refer to the same existing file
/// (same path, relative/absolute, symlink, or hard link).
/// A nonexistent path is never the same file.
fn same_file(a: &str, b: &str) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        if let (Ok(ma), Ok(mb)) = (std::fs::metadata(a), std::fs::metadata(b)) {
            return ma.dev() == mb.dev() && ma.ino() == mb.ino();
        }
        false
    }
    #[cfg(not(unix))]
    {
        match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
            (Ok(pa), Ok(pb)) => pa == pb,
            _ => false,
        }
    }
}

fn parse_size(v: &str) -> Result<RectSize, String> {
    let bad = || format!("invalid --size: {v:?} (expected WxH)");
    let (w, h) = v.split_once('x').ok_or_else(bad)?;
    Ok(RectSize {
        width: w.parse().map_err(|_| bad())?,
        height: h.parse().map_err(|_| bad())?,
    })
}

fn parse_corners(v: &str) -> Result<[Point2d; 4], String> {
    let nums = v
        .split(',')
        .map(|s| s.trim().parse::<f64>())
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| format!("invalid --corners: {v:?} (expected 8 numbers)"))?;
    if nums.len() != 8 {
        return Err(format!(
            "invalid --corners: expected 8 numbers, got {}",
            nums.len()
        ));
    }
    Ok([0, 1, 2, 3].map(|i| Point2d::new(nums[2 * i], nums[2 * i + 1])))
}

fn run() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    let mut paths = Vec::new();
    let mut size = None;
    let mut corners = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--size" => size = Some(parse_size(&args.next().ok_or("--size needs a value")?)?),
            "--corners" => {
                corners = Some(parse_corners(
                    &args.next().ok_or("--corners needs a value")?,
                )?)
            }
            _ => paths.push(a),
        }
    }
    let ([input, output], Some(size), Some(corners)) = (paths.as_slice(), size, corners) else {
        return Err(USAGE.into());
    };
    if same_file(input, output) {
        return Err(format!(
            "input and output are the same file: {input} (refusing to overwrite input)"
        ));
    }
    let h = rectify_matrix(corners, size).map_err(|e| e.to_string())?;
    let src = imgcodecs::imread(input, imgcodecs::IMREAD_COLOR)
        .map_err(|e| format!("cannot read {input}: {e}"))?;
    if src.empty() {
        return Err(format!("cannot read image: {input}"));
    }
    let dst = warp_to_rect(&src, &h, size).map_err(|e| e.to_string())?;
    match imgcodecs::imwrite(output, &dst, &Vector::new()) {
        Ok(true) => {}
        Ok(false) => return Err(format!("cannot write {output}")),
        Err(e) => return Err(format!("cannot write {output}: {e}")),
    }
    for row in h.matrix() {
        eprintln!("{:.9} {:.9} {:.9}", row[0], row[1], row[2]);
    }
    Ok(output.clone())
}

fn main() -> ExitCode {
    match run() {
        Ok(p) => {
            println!("{p}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
