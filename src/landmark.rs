//! Hand landmark inference via ONNX Runtime (`ort`).
//!
//! Uses a MediaPipe Hands landmark model converted to ONNX
//! (input: `[N, 3, 224, 224]` RGB, 0.0-1.0). The model outputs 21 hand
//! landmarks in the 224x224 input pixel space, which are scaled back to the
//! original image size.

use anyhow::{anyhow, bail, Result};
use ndarray::Array4;
use ort::session::Session;
use ort::value::TensorRef;
use std::path::Path;

/// Collapses an `ort` error (which can carry a non-`Send` recovery context that
/// `anyhow` refuses) into a plain `anyhow::Error` via its `Display`.
fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow!("ort: {e}")
}

/// Model input is a square of this size (pixels).
pub const INPUT_SIZE: u32 = 224;
/// MediaPipe hand topology: 21 landmarks per hand.
pub const NUM_LANDMARKS: usize = 21;
/// Landmark index of the index-finger tip in the MediaPipe hand topology.
pub const INDEX_FINGER_TIP: usize = 8;

/// Human-readable names for the 21 MediaPipe hand landmarks.
pub const LANDMARK_NAMES: [&str; NUM_LANDMARKS] = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
];

#[derive(Debug, Clone, Copy)]
pub struct Landmark {
    /// X in original image pixels.
    pub x: f32,
    /// Y in original image pixels.
    pub y: f32,
    /// Relative depth (model units; smaller is closer to the camera).
    pub z: f32,
}

#[derive(Debug)]
pub struct HandResult {
    pub landmarks: [Landmark; NUM_LANDMARKS],
    /// Confidence that a hand is present (0.0-1.0).
    pub score: f32,
    /// 0.0 = left hand, 1.0 = right hand.
    pub handedness: f32,
}

pub struct HandLandmarker {
    session: Session,
}

impl HandLandmarker {
    pub fn from_file(model_path: &Path) -> Result<Self> {
        let mut builder = Session::builder().map_err(ort_err)?;
        let session = builder
            .commit_from_file(model_path)
            .map_err(|e| ort_err(e).context(format!("loading model {}", model_path.display())))?;
        Ok(Self { session })
    }

    /// Names and shapes of the model inputs/outputs, for diagnostics.
    pub fn io_summary(&self) -> String {
        let inputs: Vec<String> = self
            .session
            .inputs()
            .iter()
            .map(|i| format!("{} {:?}", i.name(), i.dtype()))
            .collect();
        let outputs: Vec<String> = self
            .session
            .outputs()
            .iter()
            .map(|o| format!("{} {:?}", o.name(), o.dtype()))
            .collect();
        format!("inputs: [{}]\noutputs: [{}]", inputs.join(", "), outputs.join(", "))
    }

    /// Runs inference on an already-preprocessed `[1, 3, 224, 224]` RGB tensor.
    ///
    /// `orig_w`/`orig_h` are the source image dimensions, used to map the
    /// model's 224x224-space landmarks back to source pixels. This is the shared
    /// core used by both the `image`-crate and OpenCV frame paths.
    pub fn detect_input(
        &mut self,
        input: &Array4<f32>,
        orig_w: f32,
        orig_h: f32,
    ) -> Result<HandResult> {
        let input_name = self.session.inputs()[0].name().to_string();
        let tensor = TensorRef::from_array_view(input).map_err(ort_err)?;
        let outputs = self
            .session
            .run(ort::inputs![input_name.as_str() => tensor])
            .map_err(ort_err)?;

        let mut landmarks_raw: Option<Vec<f32>> = None;
        let mut score = f32::NAN;
        let mut handedness = f32::NAN;
        for (name, value) in outputs.iter() {
            let (_, data) = value.try_extract_tensor::<f32>().map_err(ort_err)?;
            if data.len() == NUM_LANDMARKS * 3 {
                landmarks_raw = Some(data.to_vec());
            } else if name.contains("score") {
                score = data[0];
            } else if name.contains("left") || name.contains("right") || name.contains("hand") {
                handedness = data[0];
            }
        }
        let Some(raw) = landmarks_raw else {
            bail!(
                "no output with {} values (21 landmarks x 3) found",
                NUM_LANDMARKS * 3
            );
        };

        // Model coordinates live in the 224x224 input space; map back to the
        // original image size.
        let (sx, sy) = (orig_w / INPUT_SIZE as f32, orig_h / INPUT_SIZE as f32);
        let landmarks = std::array::from_fn(|i| Landmark {
            x: raw[i * 3] * sx,
            y: raw[i * 3 + 1] * sy,
            z: raw[i * 3 + 2],
        });

        Ok(HandResult {
            landmarks,
            score,
            handedness,
        })
    }
}
