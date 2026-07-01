//! Drackpad — Milestone 0/1 spike.
//!
//! Loads a MediaPipe hand-landmark ONNX model and reports the index-finger tip
//! coordinates. Frames come from OpenCV: either still image files or a live
//! camera.
//!
//! Usage:
//!   drackpad [MODEL.onnx] [IMAGE ...]     # run on image files (via cv::imread)
//!   drackpad --camera [INDEX] [FRAMES]    # grab from a camera and detect
//!
//! Defaults to the bundled model and the images under `testdata/`.

mod capture;
mod landmark;

use anyhow::Result;
use landmark::{HandLandmarker, HandResult, INDEX_FINGER_TIP, LANDMARK_NAMES};
use std::path::{Path, PathBuf};

const DEFAULT_MODEL: &str = "models/hand_landmark_sparse_Nx3x224x224.onnx";

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();

    if let Some(pos) = args.iter().position(|a| a == "--camera") {
        return run_camera(&args, pos);
    }

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

/// Loads an image via OpenCV, runs detection, and prints all 21 landmarks.
fn run_one(landmarker: &mut HandLandmarker, path: &Path) -> Result<()> {
    let bgr = capture::imread_bgr(&path.to_string_lossy())?;
    let (input, w, h) = capture::mat_to_input(&bgr)?;
    let result = landmarker.detect_input(&input, w as f32, h as f32)?;

    print_result(&format!("{}", path.display()), w, h, &result);

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

/// `--camera [INDEX] [FRAMES]`: grab frames from a camera and detect on each.
fn run_camera(args: &[String], flag_pos: usize) -> Result<()> {
    // Positional args after `--camera`: optional device index, optional frame count.
    let rest: Vec<&String> = args[flag_pos + 1..].iter().collect();
    let index: i32 = rest.first().and_then(|s| s.parse().ok()).unwrap_or(0);
    let frames: u32 = rest.get(1).and_then(|s| s.parse().ok()).unwrap_or(30);

    let model_path = args
        .iter()
        .find(|a| a.ends_with(".onnx"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL));

    println!("Loading model: {}", model_path.display());
    let mut landmarker = HandLandmarker::from_file(&model_path)?;

    println!("Opening camera index {index} for {frames} frames...");
    let mut cam = capture::Camera::open(index)?;

    for n in 0..frames {
        let Some(frame) = cam.read()? else {
            println!("camera stream ended after {n} frames");
            break;
        };
        let (input, w, h) = capture::mat_to_input(&frame)?;
        let result = landmarker.detect_input(&input, w as f32, h as f32)?;
        print_result(&format!("frame {n}"), w, h, &result);
    }
    Ok(())
}

fn print_result(label: &str, w: u32, h: u32, result: &HandResult) {
    let tip = result.landmarks[INDEX_FINGER_TIP];
    println!(
        "[{label}] {w}x{h}  score={:.3} handedness={:.3}",
        result.score, result.handedness
    );
    println!(
        "    {} (index_finger_tip): x={:.1} y={:.1} z={:.3}",
        INDEX_FINGER_TIP, tip.x, tip.y, tip.z
    );
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
