use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest_dir.parent().unwrap().parent().unwrap();
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    let nvcc = env::var("NVCC").unwrap_or_else(|_| {
        if PathBuf::from("/opt/cuda/bin/nvcc").exists() {
            "/opt/cuda/bin/nvcc".to_string()
        } else {
            "nvcc".to_string()
        }
    });
    let arch = env::var("M3TIS_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());

    for name in ["router_remap", "exl3_moe", "linear_q4", "glm_ops", "dflash2", "exl3_requant"] {
        let src = root.join(format!("kernels/{name}.cu"));
        let cubin = out_dir.join(format!("{name}.cubin"));
        println!("cargo:rerun-if-changed={}", src.display());

        let status = Command::new(&nvcc)
            .args([
                &format!("-arch={arch}"),
                "-O3",
                "--use_fast_math",
                "-cubin",
                src.to_str().unwrap(),
                "-o",
                cubin.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to invoke nvcc ({nvcc}): {e}"));
        assert!(status.success(), "nvcc failed compiling {}", src.display());
    }
}
