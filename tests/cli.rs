use std::path::PathBuf;
use std::process::{Command, Output};

const BIN: &str = env!("CARGO_BIN_EXE_smooth-csv");
const STEP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/testdata/smoothing/step.csv");
const HEADER: &str = "frame,raw_x,raw_y,smooth_x,smooth_y";

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("drackpad-smooth-csv-{}-{name}", std::process::id()))
}
fn run(args: &[&str]) -> Output {
    Command::new(BIN).args(args).output().unwrap()
}

#[test]
fn step_with_alpha_and_default() {
    let a = tmp("a.csv");
    let b = tmp("b.csv");
    let o = run(&["--alpha", "0.5", STEP, a.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&o.stdout),
        format!("{}\n", a.display())
    );
    let o2 = run(&[STEP, b.to_str().unwrap()]);
    assert_eq!(o2.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&o2.stdout),
        format!("{}\n", b.display())
    );
    let ta = std::fs::read_to_string(&a).unwrap();
    assert!(!ta.is_empty());
    assert_eq!(ta.lines().next().unwrap(), HEADER);
    assert_eq!(ta.lines().count(), 11);
    assert!(ta.lines().any(|l| l == "2,100,0,50,0"));
    assert_eq!(ta, std::fs::read_to_string(&b).unwrap());
    let _ = std::fs::remove_file(a);
    let _ = std::fs::remove_file(b);
}

#[test]
fn failures_are_nonzero_with_stderr() {
    let out = tmp("never.csv");
    let o = out.to_str().unwrap();
    let cases: Vec<Vec<&str>> = vec![
        vec!["--alpha", "0", STEP, o],
        vec!["--alpha", "abc", STEP, o],
        vec!["/nonexistent/in.csv", o],
        vec![STEP],
        vec![],
    ];
    for args in cases {
        let r = run(&args);
        assert_ne!(r.status.code(), Some(0), "{args:?}");
        assert!(!r.stderr.is_empty(), "{args:?}");
        assert!(r.stdout.is_empty(), "{args:?}");
    }
    assert!(!out.exists());
}

#[test]
fn empty_input_fails_without_output() {
    let i = tmp("empty-in.csv");
    let o = tmp("empty-out.csv");
    std::fs::write(&i, "").unwrap();
    let r = run(&[i.to_str().unwrap(), o.to_str().unwrap()]);
    assert_eq!(r.status.code(), Some(1));
    let err = String::from_utf8_lossy(&r.stderr);
    assert!(err.contains("line 1"), "{err}");
    assert!(!o.exists());
    let _ = std::fs::remove_file(i);
}

#[test]
fn header_only_input() {
    let i = tmp("hdr-in.csv");
    let o = tmp("hdr-out.csv");
    std::fs::write(&i, "frame,x,y\n").unwrap();
    let r = run(&[i.to_str().unwrap(), o.to_str().unwrap()]);
    assert_eq!(r.status.code(), Some(0));
    assert_eq!(std::fs::read_to_string(&o).unwrap(), format!("{HEADER}\n"));
    let _ = std::fs::remove_file(i);
    let _ = std::fs::remove_file(o);
}

fn rows_with_alpha(alpha: &str, name: &str) -> Vec<String> {
    let o = tmp(name);
    let r = run(&["--alpha", alpha, STEP, o.to_str().unwrap()]);
    assert_eq!(r.status.code(), Some(0));
    assert_eq!(
        String::from_utf8_lossy(&r.stdout),
        format!("{}\n", o.display())
    );
    let t = std::fs::read_to_string(&o).unwrap();
    let _ = std::fs::remove_file(o);
    t.lines().map(String::from).collect()
}

#[test]
fn alpha_quarter_changes_output_by_hand_computed_values() {
    // s2 = 0.25*100 + 0.75*0 = 25; s3 = 0.25*100 + 0.75*25 = 43.75
    let rows = rows_with_alpha("0.25", "q.csv");
    assert_eq!(rows[0], HEADER);
    assert_eq!(rows[3], "2,100,0,25,0");
    assert_eq!(rows[4], "3,100,0,43.75,0");
}

#[test]
fn alpha_one_means_no_smoothing() {
    let rows = rows_with_alpha("1", "one.csv");
    assert_eq!(rows[2], "1,0,0,0,0");
    assert_eq!(rows[3], "2,100,0,100,0");
    for r in &rows[1..] {
        let c: Vec<&str> = r.split(',').collect();
        assert_eq!((c[1], c[2]), (c[3], c[4]), "{r}");
    }
}
