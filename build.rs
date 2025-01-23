fn main() {
    #[cfg(feature = "gpu")]
    build_cuda_libs();
}

#[cfg(feature = "gpu")]
fn get_compute_capability() -> Option<(i32, i32)> {
    use std::process::Command;

    // Try to run nvidia-smi to get device info
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap_major,compute_cap_minor", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let output_str = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = output_str.trim().split(',').collect();
    if parts.len() != 2 {
        return None;
    }

    let major = parts[0].trim().parse::<i32>().ok()?;
    let minor = parts[1].trim().parse::<i32>().ok()?;

    Some((major, minor))
}

#[cfg(feature = "gpu")]
fn build_cuda_libs() {
    println!("cargo::rerun-if-changed=kernels/");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("kernels/utils.cu")
        .file("kernels/vanity.cu")
        .file("kernels/base58.cu")
        .file("kernels/sha256.cu")
        .flag("-cudart=static");

    // Get compute capability or fallback to default
    if let Some((major, minor)) = get_compute_capability() {
        let arch = format!("compute_{}{}", major, minor);
        let code = format!("sm_{}{}", major, minor);
        println!("Detected GPU compute capability: {}.{}", major, minor);
        build
            .flag(&format!("-gencode=arch={},code={}", arch, arch))
            .flag(&format!("-gencode=arch={},code={}", arch, code));
    } else {
        // Fallback to a safe default that supports most modern GPUs
        println!("Could not detect GPU compute capability, using default (compute_86)");
        build
            .flag("-gencode=arch=compute_86,code=sm_86")
            .flag("-gencode=arch=compute_86,code=compute_86");
    }

    build.compile("libvanity.a");

    // Add link directory
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Emit the location of the compiled library
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
