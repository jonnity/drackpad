use opencv::core::{CV_8UC3, Mat, Point, Scalar, Vector};
use opencv::prelude::*;
use opencv::{imgcodecs, imgproc};
use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_rectify");
const CORNERS: &str = "120,40,280,40,400,200,0,200";

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("drackpad-rectify-{}-{name}", std::process::id()))
}

fn make_input(name: &str) -> PathBuf {
    let path = tmp(name);
    let mut img = Mat::new_rows_cols_with_default(300, 400, CV_8UC3, Scalar::all(0.0)).unwrap();
    let pts: Vector<Point> = [(120, 40), (280, 40), (400, 200), (0, 200)]
        .iter()
        .map(|&(x, y)| Point::new(x, y))
        .collect();
    imgproc::fill_convex_poly(&mut img, &pts, Scalar::all(255.0), imgproc::LINE_8, 0).unwrap();
    assert!(imgcodecs::imwrite(path.to_str().unwrap(), &img, &Vector::new()).unwrap());
    path
}

fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

fn s(p: &std::path::Path) -> &str {
    p.to_str().unwrap()
}

fn assert_failed(o: &Output) {
    assert_eq!(o.status.code(), Some(1));
    assert!(o.stdout.is_empty());
    assert!(!o.stderr.is_empty());
}

#[test]
fn c1_success() {
    let input = make_input("c1-in.png");
    let output = tmp("c1-out.png");
    let o = run(&[
        s(&input),
        s(&output),
        "--size",
        "200x100",
        "--corners",
        CORNERS,
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert_eq!(
        String::from_utf8(o.stdout).unwrap(),
        format!("{}\n", s(&output))
    );
    assert!(std::fs::metadata(&output).unwrap().len() > 0);
    let img = imgcodecs::imread(s(&output), imgcodecs::IMREAD_COLOR).unwrap();
    assert_eq!((img.cols(), img.rows()), (200, 100));
    for y in 3..97 {
        for x in 3..197 {
            let v = img.at_2d::<opencv::core::Vec3b>(y, x).unwrap();
            assert!(v[0] >= 250, "({x},{y}) = {v:?}");
        }
    }
    let _ = std::fs::remove_file(&input);
    let _ = std::fs::remove_file(&output);
}

#[test]
fn c2_collinear() {
    let input = make_input("c2-in.png");
    let output = tmp("c2-out.png");
    let o = run(&[
        s(&input),
        s(&output),
        "--size",
        "200x100",
        "--corners",
        "0,0,50,0,100,0,0,50",
    ]);
    assert_failed(&o);
    assert!(String::from_utf8_lossy(&o.stderr).contains("collinear"));
    assert!(!output.exists());
    let _ = std::fs::remove_file(&input);
}

#[test]
fn c3_bad_args() {
    let input = make_input("c3-in.png");
    let output = tmp("c3-out.png");
    let (i, o) = (s(&input), s(&output));
    let cases: Vec<Vec<&str>> = vec![
        vec![i, o],
        vec![i, "--size", "200x100", "--corners", CORNERS],
        vec![i, o, "--size", "0x10", "--corners", CORNERS],
        vec![i, o, "--size", "1x10", "--corners", CORNERS],
        vec![i, o, "--size", "abc", "--corners", CORNERS],
        vec![i, o, "--size", "200x100", "--corners", "1,2,3,4,5,6,7"],
        vec![i, o, "--size", "200x100", "--corners", "1,2,3,4,5,6,7,x"],
        vec![i, o, "--size", "200x100"],
        vec![i, o, "--corners", CORNERS],
    ];
    for args in cases {
        let out = run(&args);
        assert_failed(&out);
        assert!(!output.exists(), "{args:?}");
    }
    let _ = std::fs::remove_file(&input);
}

#[test]
fn c4_missing_input() {
    let input = tmp("c4-missing.png");
    let output = tmp("c4-out.png");
    let o = run(&[
        s(&input),
        s(&output),
        "--size",
        "200x100",
        "--corners",
        CORNERS,
    ]);
    assert_failed(&o);
    assert!(!output.exists());
}

#[test]
fn c5_same_file() {
    let input = make_input("c5-in.png");
    let before = std::fs::read(&input).unwrap();
    let dotted = format!(
        "{}/./{}",
        input.parent().unwrap().display(),
        input.file_name().unwrap().to_str().unwrap()
    );
    for other in [s(&input).to_string(), dotted] {
        let o = run(&[s(&input), &other, "--size", "200x100", "--corners", CORNERS]);
        assert_failed(&o);
        assert_eq!(std::fs::read(&input).unwrap(), before);
    }
    let _ = std::fs::remove_file(&input);
}

#[test]
fn c6_size_overflow() {
    let input = make_input("c6-in.png");
    let output = tmp("c6-out.png");
    for size in ["4294967295x100", "2147483648x100", "100x4294967295"] {
        let o = run(&[s(&input), s(&output), "--size", size, "--corners", CORNERS]);
        assert_failed(&o);
        assert!(
            String::from_utf8_lossy(&o.stderr).contains("too large"),
            "{size}"
        );
        assert!(!output.exists(), "{size}");
    }
    let _ = std::fs::remove_file(&input);
}

#[test]
fn c7_f32_collapsed_corners() {
    let input = make_input("c7-in.png");
    let output = tmp("c7-out.png");
    let o = run(&[
        s(&input),
        s(&output),
        "--size",
        "100x100",
        "--corners",
        "100000000,100000000,100000001,100000000,100000001,100000001,100000000,100000001.5",
    ]);
    assert_failed(&o);
    assert!(String::from_utf8_lossy(&o.stderr).contains("duplicated"));
    assert!(!output.exists());
    let _ = std::fs::remove_file(&input);
}
