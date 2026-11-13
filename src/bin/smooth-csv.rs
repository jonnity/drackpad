//! Usage: smooth-csv [--alpha <A>] <INPUT.csv> <OUTPUT.csv>
//!
//! Reads `frame,x,y` and writes `frame,raw_x,raw_y,smooth_x,smooth_y`
//! with an exponential moving average (default alpha 0.5).

use drackpad::smoothing::{DEFAULT_ALPHA, Ema};
use drackpad::trace_csv::compare_csv;
use std::process::ExitCode;

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

fn run() -> Result<String, String> {
    let mut args = std::env::args().skip(1);
    let mut alpha = DEFAULT_ALPHA;
    let mut paths = Vec::new();
    while let Some(a) = args.next() {
        if a == "--alpha" {
            let v = args.next().ok_or("--alpha needs a value")?;
            alpha = v.parse().map_err(|_| format!("invalid alpha: {v:?}"))?;
        } else {
            paths.push(a);
        }
    }
    let [input, output] = paths.as_slice() else {
        return Err("usage: smooth-csv [--alpha <A>] <INPUT.csv> <OUTPUT.csv>".into());
    };
    if same_file(input, output) {
        return Err(format!(
            "input and output are the same file: {input} (refusing to overwrite input)"
        ));
    }
    let mut ema = Ema::new(alpha).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(input).map_err(|e| format!("cannot read {input}: {e}"))?;
    let out = compare_csv(&text, &mut ema).map_err(|e| format!("{input}: {e}"))?;
    std::fs::write(output, out).map_err(|e| format!("cannot write {output}: {e}"))?;
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
