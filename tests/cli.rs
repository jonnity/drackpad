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

fn assert_rejected_and_untouched(args: &[&str], input: &std::path::Path, original: &str) {
    let r = run(args);
    assert_ne!(r.status.code(), Some(0), "{args:?}");
    assert!(!r.stderr.is_empty(), "{args:?}");
    assert!(r.stdout.is_empty(), "{args:?}");
    assert_eq!(std::fs::read_to_string(input).unwrap(), original);
}

const ORIGINAL: &str = "frame,x,y\n0,0,0\n1,10,10\n";

#[test]
fn same_path_string_is_rejected() {
    let i = tmp("same-str.csv");
    std::fs::write(&i, ORIGINAL).unwrap();
    let s = i.to_str().unwrap();
    assert_rejected_and_untouched(&[s, s], &i, ORIGINAL);
    let _ = std::fs::remove_file(i);
}

#[test]
fn same_file_via_dot_slash_and_relative_is_rejected() {
    let i = tmp("same-rel.csv");
    std::fs::write(&i, ORIGINAL).unwrap();
    let dir = i.parent().unwrap();
    let name = i.file_name().unwrap().to_str().unwrap();
    let abs = i.to_str().unwrap();
    let dotted = format!("{}/./{name}", dir.display());
    assert_rejected_and_untouched(&[abs, &dotted], &i, ORIGINAL);
    // relative input vs absolute output, run from the file's directory
    for (a, b) in [(format!("./{name}"), abs), (name.to_string(), abs)] {
        let r = Command::new(BIN)
            .current_dir(dir)
            .args([a.as_str(), b])
            .output()
            .unwrap();
        assert_ne!(r.status.code(), Some(0), "{a} {b}");
        assert!(!r.stderr.is_empty());
        assert!(r.stdout.is_empty());
        assert_eq!(std::fs::read_to_string(&i).unwrap(), ORIGINAL);
    }
    let _ = std::fs::remove_file(i);
}

#[cfg(unix)]
#[test]
fn symlink_and_hardlink_to_input_are_rejected() {
    let i = tmp("same-link.csv");
    let sym = tmp("same-link-sym.csv");
    let hard = tmp("same-link-hard.csv");
    for p in [&sym, &hard] {
        let _ = std::fs::remove_file(p);
    }
    std::fs::write(&i, ORIGINAL).unwrap();
    std::os::unix::fs::symlink(&i, &sym).unwrap();
    std::fs::hard_link(&i, &hard).unwrap();
    let s = i.to_str().unwrap();
    assert_rejected_and_untouched(&[s, sym.to_str().unwrap()], &i, ORIGINAL);
    assert_rejected_and_untouched(&[sym.to_str().unwrap(), s], &i, ORIGINAL);
    assert_rejected_and_untouched(&[s, hard.to_str().unwrap()], &i, ORIGINAL);
    for p in [i, sym, hard] {
        let _ = std::fs::remove_file(p);
    }
}

// Not-the-same-file cases: must succeed (no false positives).
const SMOOTHED: &str = "frame,raw_x,raw_y,smooth_x,smooth_y\n0,0,0,0,0\n1,10,10,5,5\n";

fn assert_ok_with(r: &Output, expected_stdout: &str) {
    assert_eq!(
        r.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&r.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&r.stdout), expected_stdout);
}

#[test]
fn existing_different_output_is_overwritten() {
    let i = tmp("diff-in.csv");
    let o = tmp("diff-out.csv"); // same directory / filesystem as the input
    std::fs::write(&i, ORIGINAL).unwrap();
    std::fs::write(&o, "old unrelated content\n").unwrap();
    let r = run(&["--alpha", "0.5", i.to_str().unwrap(), o.to_str().unwrap()]);
    assert_ok_with(&r, &format!("{}\n", o.display()));
    assert_eq!(std::fs::read_to_string(&o).unwrap(), SMOOTHED);
    assert_eq!(std::fs::read_to_string(&i).unwrap(), ORIGINAL);
    let _ = std::fs::remove_file(i);
    let _ = std::fs::remove_file(o);
}

#[test]
fn similar_names_and_relative_paths_in_same_dir_succeed() {
    let i = tmp("sim.csv");
    let o = tmp("sim.csv.out");
    std::fs::write(&i, ORIGINAL).unwrap();
    std::fs::write(&o, "old\n").unwrap();
    let dir = i.parent().unwrap();
    let a = i.file_name().unwrap().to_str().unwrap();
    let b = o.file_name().unwrap().to_str().unwrap();
    // one side relative, the other absolute / with ./ prefix
    let rel_in = format!("./{a}");
    let abs_out = o.to_str().unwrap().to_string();
    let rel_out = format!("./{b}");
    for (x, y) in [(rel_in.as_str(), abs_out.as_str()), (a, rel_out.as_str())] {
        std::fs::write(&o, "old\n").unwrap();
        let r = Command::new(BIN)
            .current_dir(dir)
            .args([x, y])
            .output()
            .unwrap();
        assert_ok_with(&r, &format!("{y}\n"));
        assert_eq!(std::fs::read_to_string(&o).unwrap(), SMOOTHED);
        assert_eq!(std::fs::read_to_string(&i).unwrap(), ORIGINAL);
    }
    let _ = std::fs::remove_file(i);
    let _ = std::fs::remove_file(o);
}

#[cfg(unix)]
#[test]
fn symlink_to_different_file_as_output_writes_through_link() {
    let i = tmp("lnk-in.csv");
    let target = tmp("lnk-target.csv");
    let link = tmp("lnk-link.csv");
    let _ = std::fs::remove_file(&link);
    std::fs::write(&i, ORIGINAL).unwrap();
    std::fs::write(&target, "old target\n").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();
    let r = run(&[i.to_str().unwrap(), link.to_str().unwrap()]);
    assert_ok_with(&r, &format!("{}\n", link.display()));
    assert_eq!(std::fs::read_to_string(&target).unwrap(), SMOOTHED);
    assert!(
        std::fs::symlink_metadata(&link)
            .unwrap()
            .file_type()
            .is_symlink()
    );
    assert_eq!(std::fs::read_to_string(&i).unwrap(), ORIGINAL);
    for p in [i, target, link] {
        let _ = std::fs::remove_file(p);
    }
}
