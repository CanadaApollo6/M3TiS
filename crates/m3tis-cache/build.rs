use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let src = root.join("kernels/exl3_cpu.cpp");
    let obj = out_dir.join("exl3_cpu.o");
    let lib = out_dir.join("libm3tis_exl3_cpu.a");
    println!("cargo:rerun-if-changed={}", src.display());

    let cxx = env::var("CXX").unwrap_or_else(|_| "g++".to_string());
    let status = Command::new(&cxx)
        .args([
            "-O3",
            "-std=c++20",
            "-march=native",
            "-mavx2",
            "-mfma",
            "-mf16c",
            "-fno-exceptions",
            "-fno-rtti",
            "-pthread",
            "-c",
            src.to_str().unwrap(),
            "-o",
            obj.to_str().unwrap(),
        ])
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke C++ compiler ({cxx}): {e}"));
    assert!(status.success(), "C++ compilation failed for {}", src.display());

    let ar_status = Command::new("ar")
        .args(["rcs", lib.to_str().unwrap(), obj.to_str().unwrap()])
        .status()
        .expect("failed to invoke ar");
    assert!(ar_status.success(), "ar failed building {}", lib.display());

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=m3tis_exl3_cpu");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
}
