use drackpad::smoothing::Ema;
use drackpad::trace_csv::*;

const OUTLIER: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/smoothing/outlier.csv"
));
const STEP: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/smoothing/step.csv"
));
const MISSING: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/testdata/smoothing/missing.csv"
));
const HEADER: &str = "frame,raw_x,raw_y,smooth_x,smooth_y\n";

fn cmp(t: &str) -> Result<String, CsvError> {
    compare_csv(t, &mut Ema::default())
}

#[test]
fn outlier_output() {
    let exp = format!(
        "{HEADER}0,100,100,100,100\n1,100,100,100,100\n2,100,100,100,100\n3,200,100,150,100\n4,100,100,125,100\n5,100,100,112.5,100\n6,100,100,106.25,100\n7,100,100,103.125,100\n8,100,100,101.5625,100\n9,100,100,100.78125,100\n"
    );
    assert_eq!(cmp(OUTLIER).unwrap(), exp);
}

#[test]
fn step_output() {
    let out = cmp(STEP).unwrap();
    let col: Vec<&str> = out
        .lines()
        .skip(1)
        .map(|l| l.split(',').nth(3).unwrap())
        .collect();
    assert_eq!(
        col,
        [
            "0",
            "0",
            "50",
            "75",
            "87.5",
            "93.75",
            "96.875",
            "98.4375",
            "99.21875",
            "99.609375"
        ]
    );
    assert!(out.lines().any(|l| l == "6,100,0,96.875,0"));
}

#[test]
fn missing_output() {
    let out = cmp(MISSING).unwrap();
    assert!(out.lines().any(|l| l == "3,,,,"));
    assert!(out.lines().any(|l| l == "4,100,100,100,100"));
}

#[test]
fn empty_and_header_only() {
    assert_eq!(cmp("").unwrap_err().line, 1);
    assert_eq!(cmp("\n").unwrap_err().line, 1);
    for t in ["frame,x,y", "frame,x,y\n", "frame,x,y\r\n"] {
        assert_eq!(cmp(t).unwrap(), HEADER);
        assert!(parse_input(t).unwrap().is_empty());
    }
}

#[test]
fn errors_with_lines() {
    let cases = [
        ("a,b,c\n", 1),
        ("frame,x,y\n1,2\n", 2),
        ("frame,x,y\n0,1,2,3\n", 2),
        ("frame,x,y\n0,1,2\n1,2,3,\n", 3),
        ("frame,x,y\n0,1,2,\n", 2),
        ("frame,x,y\n0,1,2\n1,abc,2\n", 3),
        ("frame,x,y\n0,NaN,2\n", 2),
        ("frame,x,y\n0,1,inf\n", 2),
        ("frame,x,y\n0,1,\n", 2),
        ("frame,x,y\n0,,1\n", 2),
        ("frame,x,y\n0,1,2\n\n2,1,2\n", 3),
    ];
    for (t, line) in cases {
        assert_eq!(cmp(t).unwrap_err().line, line, "{t:?}");
    }
}

#[test]
fn crlf_input() {
    let out = cmp("frame,x,y\r\n0,1,2\r\n1,3,2\r\n").unwrap();
    assert_eq!(out, format!("{HEADER}0,1,2,1,2\n1,3,2,2,2\n"));
}

#[test]
fn moving_xy_output_matches_hand_computed_alpha_half() {
    // s = 0.5*p + 0.5*s, first sample passes through.
    let input = "frame,x,y\n0,0,0\n1,10,20\n2,10,20\n3,10,0\n";
    let exp = format!("{HEADER}0,0,0,0,0\n1,10,20,5,10\n2,10,20,7.5,15\n3,10,0,8.75,7.5\n");
    assert_eq!(cmp(input).unwrap(), exp);
}

#[test]
fn y_only_ramp_output_matches_hand_computed() {
    let input = "frame,x,y\n0,7,0\n1,7,4\n2,7,8\n";
    let exp = format!("{HEADER}0,7,0,7,0\n1,7,4,7,2\n2,7,8,7,5\n");
    assert_eq!(cmp(input).unwrap(), exp);
}

#[test]
fn moving_xy_output_matches_hand_computed_alpha_quarter() {
    // s = 0.25*p + 0.75*s
    let input = "frame,x,y\n0,0,0\n1,8,4\n2,8,4\n";
    let exp = format!("{HEADER}0,0,0,0,0\n1,8,4,2,1\n2,8,4,3.5,1.75\n");
    let mut e = Ema::new(0.25).unwrap();
    assert_eq!(compare_csv(input, &mut e).unwrap(), exp);
}
