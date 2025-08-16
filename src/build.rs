use std::process::Command;

fn main() {
    #[cfg(feature = "rocm")]
    compile_hip_kernel();
}

#[cfg(feature = "rocm")]
fn compile_hip_kernel() {
    use std::env;
    use std::path::PathBuf;
    // Validate and sanitize build environment
    let hipcc_path = match which::which("hipcc") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("hipcc not found in PATH. Aborting HIP kernel build.");
            return;
        }
    };
    let kernel_src = PathBuf::from("kernels/l2_kernel.hip");
    let kernel_out = PathBuf::from("kernels/l2_kernel.hsaco");
    if !kernel_src.exists() {
        eprintln!("Kernel source {:?} does not exist.", kernel_src);
        return;
    }
    // Only allow known safe arguments
    let args = ["--genco", "-o", kernel_out.to_str().unwrap(), kernel_src.to_str().unwrap()];
    let status = Command::new(hipcc_path)
        .args(&args)
        .env_clear()
        .envs(env::vars().filter(|(k,_)| k == "PATH" || k == "HOME"))
        .status();
    let status = match status {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to compile HIP kernel: {}", e);
            return;
        }
    };
    if !status.success() {
        eprintln!("HIP kernel compilation failed with exit code: {:?}", status.code());
        return;
    }
    println!("cargo:rerun-if-changed=kernels/l2_kernel.hip");
}