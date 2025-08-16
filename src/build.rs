use std::process::Command;

fn main() {
    #[cfg(feature = "rocm")]
    compile_hip_kernel();
}

#[cfg(feature = "rocm")]
fn compile_hip_kernel() {
    let status = Command::new("hipcc")
        .args(&[
            "--genco",
            "-o", "kernels/l2_kernel.hsaco",
            "kernels/l2_kernel.hip"
        ])
        .status();
    let status = match status {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to compile HIP kernel: {}", e);
            return;
        }
    };
    
    if !status.success() {
        panic!("HIP kernel compilation failed");
    }
    
    println!("cargo:rerun-if-changed=kernels/l2_kernel.hip");
}