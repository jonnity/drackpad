//! OpenCV-backed frame acquisition and preprocessing.
//!
//! Provides two frame sources — still images (`imread`) and a live camera
//! (`VideoCapture`) — plus the conversion from an OpenCV BGR `Mat` into the
//! `[1, 3, 224, 224]` RGB tensor the landmark model expects. Keeping the
//! preprocessing here (rather than leaning on a MediaPipe SDK) is what the spec
//! calls for: we own the pixel-format and resize steps.

use anyhow::{bail, Context, Result};
use ndarray::Array4;
use opencv::core::{Mat, Size, Vec3b};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc, videoio};

use crate::landmark::INPUT_SIZE;

/// Reads an image file into a BGR `Mat` (OpenCV's native channel order).
pub fn imread_bgr(path: &str) -> Result<Mat> {
    let mat = imgcodecs::imread(path, imgcodecs::IMREAD_COLOR)
        .with_context(|| format!("cv::imread {path}"))?;
    if mat.empty() {
        bail!("could not decode image: {path}");
    }
    Ok(mat)
}

/// Resizes a BGR `Mat` to the model input and packs it into a `[1, 3, H, W]`
/// RGB tensor normalized to 0.0-1.0. Returns the tensor and the *source* size
/// (so landmarks can be mapped back to the original pixels).
pub fn mat_to_input(bgr: &Mat) -> Result<(Array4<f32>, u32, u32)> {
    let (orig_w, orig_h) = (bgr.cols() as u32, bgr.rows() as u32);

    let mut resized = Mat::default();
    imgproc::resize(
        bgr,
        &mut resized,
        Size::new(INPUT_SIZE as i32, INPUT_SIZE as i32),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .context("cv::resize")?;

    let size = INPUT_SIZE as usize;
    let mut input = Array4::<f32>::zeros((1, 3, size, size));
    for y in 0..size {
        for x in 0..size {
            // `Vec3b` is [B, G, R]; the model wants channel-first R, G, B.
            let px = resized.at_2d::<Vec3b>(y as i32, x as i32)?;
            input[[0, 0, y, x]] = px[2] as f32 / 255.0;
            input[[0, 1, y, x]] = px[1] as f32 / 255.0;
            input[[0, 2, y, x]] = px[0] as f32 / 255.0;
        }
    }
    Ok((input, orig_w, orig_h))
}

/// A live camera frame source backed by OpenCV's `VideoCapture`.
pub struct Camera {
    cap: videoio::VideoCapture,
}

impl Camera {
    /// Opens the camera at the given device index (0 is usually the default).
    pub fn open(index: i32) -> Result<Self> {
        let cap = videoio::VideoCapture::new(index, videoio::CAP_ANY)
            .with_context(|| format!("opening camera index {index}"))?;
        if !cap.is_opened()? {
            bail!("could not open camera index {index} (no device / no permission)");
        }
        Ok(Self { cap })
    }

    /// Grabs the next frame as a BGR `Mat`. Returns `None` if the stream ended.
    pub fn read(&mut self) -> Result<Option<Mat>> {
        let mut frame = Mat::default();
        if !self.cap.read(&mut frame)? || frame.empty() {
            return Ok(None);
        }
        Ok(Some(frame))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opencv::core::{Scalar, CV_8UC3};

    #[test]
    fn mat_to_input_shape_range_and_channel_order() -> Result<()> {
        // A solid 10x20 BGR image with B=10, G=128, R=250.
        let bgr = Mat::new_rows_cols_with_default(
            20,
            10,
            CV_8UC3,
            Scalar::new(10.0, 128.0, 250.0, 0.0),
        )?;

        let (input, w, h) = mat_to_input(&bgr)?;

        assert_eq!((w, h), (10, 20));
        assert_eq!(input.shape(), &[1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize]);
        assert!(input.iter().all(|&v| (0.0..=1.0).contains(&v)));
        // Channels were reordered BGR -> RGB: R (250) > G (128) > B (10).
        assert!(input[[0, 0, 0, 0]] > input[[0, 1, 0, 0]]);
        assert!(input[[0, 1, 0, 0]] > input[[0, 2, 0, 0]]);
        Ok(())
    }
}
