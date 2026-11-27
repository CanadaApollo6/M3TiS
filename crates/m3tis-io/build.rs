use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest.join("csrc/uring_shim.c");
    println!("cargo:rerun-if-changed={}", src.display());

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let obj = out_dir.join("uring_shim.o");
    let lib = out_dir.join("libm3tis_uring_shim.a");

    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let st = Command::new(&cc)
        .args(["-O3", "-fPIC", "-c", src.to_str().unwrap(), "-o", obj.to_str().unwrap()])
        .status()
        .expect("failed to run cc for uring_shim.c");
    assert!(st.success(), "cc failed on uring_shim.c");

    let st = Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("failed to run ar");
    assert!(st.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=m3tis_uring_shim");
    println!("cargo:rustc-link-lib=dylib=uring");
}
