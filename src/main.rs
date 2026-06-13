//! Drackpad — Milestone 0 spike.
//!
//! Loads a MediaPipe hand-landmark ONNX model and prints the index-finger tip
//! coordinates for one or more input images.
//!
//! Usage:
//!   drackpad [MODEL.onnx] [IMAGE ...]
//!
//! Defaults to the bundled model and the images under `testdata/`.

mod landmark;

use anyhow::{Context, Result};
use landmark::{HandLandmarker, INDEX_FINGER_TIP, LANDMARK_NAMES};
use std::path::{Path, PathBuf};

const DEFAULT_MODEL: &str = "models/hand_landmark_sparse_Nx3x224x224.onnx";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    let (model_path, image_paths) = parse_args(&args);

    println!("Loading model: {}", model_path.display());
    let mut landmarker = HandLandmarker::from_file(&model_path)?;
    println!("{}\n", landmarker.io_summary());

    if image_paths.is_empty() {
        println!("No input images found. Pass image paths or add files under testdata/.");
        return Ok(());
    }

    for path in &image_paths {
        match run_one(&mut landmarker, path) {
            Ok(()) => {}
            Err(e) => eprintln!("[{}] error: {e:#}", path.display()),
        }
    }

    Ok(())
}

fn run_one(landmarker: &mut HandLandmarker, path: &Path) -> Result<()> {
    let image = image::open(path).with_context(|| format!("opening {}", path.display()))?;
    let result = landmarker.detect(&image)?;

    let tip = result.landmarks[INDEX_FINGER_TIP];
    println!(
        "[{}] {}x{}  score={:.3} handedness={:.3}",
        path.display(),
        image.width(),
        image.height(),
        result.score,
        result.handedness
    );
    println!(
        "    {} (index_finger_tip): x={:.1} y={:.1} z={:.3}",
        INDEX_FINGER_TIP, tip.x, tip.y, tip.z
    );

    // Also dump all landmarks for inspection during the spike.
    for (i, lm) in result.landmarks.iter().enumerate() {
        println!(
            "      [{:>2}] {:<18} x={:7.1} y={:7.1} z={:7.3}",
            i, LANDMARK_NAMES[i], lm.x, lm.y, lm.z
        );
    }
    println!();
    Ok(())
}

/// Splits CLI args into a model path and a list of image paths.
///
/// An argument ending in `.onnx` is treated as the model; everything else is an
/// image. Falls back to the bundled model and `testdata/` images.
fn parse_args(args: &[String]) -> (PathBuf, Vec<PathBuf>) {
    let mut model_path: Option<PathBuf> = None;
    let mut image_paths: Vec<PathBuf> = Vec::new();

    for arg in args {
        if arg.ends_with(".onnx") {
            model_path = Some(PathBuf::from(arg));
        } else {
            image_paths.push(PathBuf::from(arg));
        }
    }

    let model_path = model_path.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));
    if image_paths.is_empty() {
        image_paths = default_test_images();
    }

    (model_path, image_paths)
}

fn default_test_images() -> Vec<PathBuf> {
    let dir = Path::new("testdata");
    let mut images = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let is_image = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| matches!(e.to_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                .unwrap_or(false);
            if is_image {
                images.push(path);
            }
        }
    }
    images.sort();
    images
}
