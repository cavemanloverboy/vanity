fn main() {
    #[cfg(feature = "gpu")]
    build_cuda_libs();
}

#[cfg(feature = "gpu")]
fn get_compute_capability() -> Option<(i32, i32)> {
    use std::process::Command;

    // Try to get GPU name first
    let nvidia_smi = Command::new("nvidia-smi")
        .args(["--query-gpu=gpu_name", "--format=csv,noheader"])
        .output();

    match &nvidia_smi {
        Ok(output) if output.status.success() => {
            let gpu_name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            println!("cargo:warning=Found GPU: {}", gpu_name);

            // Map GPU series to compute capabilities
            let gpu_name = gpu_name.to_lowercase();

            // RTX 40 series (Ada Lovelace) - Compute 8.9
            if
                gpu_name.contains("40") &&
                (gpu_name.contains("4090") ||
                    gpu_name.contains("4080") ||
                    gpu_name.contains("4070") ||
                    gpu_name.contains("4060"))
            {
                println!("cargo:warning=RTX 40 series detected, using compute_89");
                return Some((8, 9));
            }

            // RTX 30 series (Ampere) - Compute 8.6
            if
                gpu_name.contains("30") &&
                (gpu_name.contains("3090") ||
                    gpu_name.contains("3080") ||
                    gpu_name.contains("3070") ||
                    gpu_name.contains("3060") ||
                    gpu_name.contains("3050"))
            {
                println!("cargo:warning=RTX 30 series detected, using compute_86");
                return Some((8, 6));
            }

            // RTX 20 series (Turing) - Compute 7.5
            if
                gpu_name.contains("20") &&
                (gpu_name.contains("2080") ||
                    gpu_name.contains("2070") ||
                    gpu_name.contains("2060"))
            {
                println!("cargo:warning=RTX 20 series detected, using compute_75");
                return Some((7, 5));
            }

            // GTX 16 series (Turing) - Compute 7.5
            if gpu_name.contains("16") && (gpu_name.contains("1660") || gpu_name.contains("1650")) {
                println!("cargo:warning=GTX 16 series detected, using compute_75");
                return Some((7, 5));
            }

            // Quadro/RTX Professional - Use latest supported
            if gpu_name.contains("quadro") || gpu_name.contains("rtx") {
                println!("cargo:warning=Professional GPU detected, using compute_86");
                return Some((8, 6));
            }
        }
        _ => {}
    }

    // Try nvcc as fallback
    let nvcc = Command::new("nvcc").args(["--version"]).output();

    match &nvcc {
        Ok(output) if output.status.success() => {
            println!("cargo:warning=Found CUDA installation, using compute_86 as safe default");
            return Some((8, 6));
        }
        _ => {}
    }

    // If both methods fail, print more detailed error
    println!("cargo:warning=Could not detect GPU compute capability:");
    if let Err(e) = &nvidia_smi {
        println!("cargo:warning=  nvidia-smi error: {}", e);
    }
    if let Err(e) = &nvcc {
        println!("cargo:warning=  nvcc error: {}", e);
    }
    println!("cargo:warning=Using compute_86 as safe default");

    Some((8, 6))
}

#[cfg(feature = "gpu")]
fn build_cuda_libs() {
    println!("cargo:rerun-if-changed=kernels/");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .file("kernels/utils.cu")
        .file("kernels/vanity.cu")
        .file("kernels/base58.cu")
        .file("kernels/sha256.cu")
        .flag("-cudart=static");

    // Get compute capability or fallback to default
    let (major, minor) = get_compute_capability().unwrap_or((8, 9));
    let arch = format!("compute_{}{}", major, minor);
    let code = format!("sm_{}{}", major, minor);

    build
        .flag(&format!("-gencode=arch={},code={}", arch, arch))
        .flag(&format!("-gencode=arch={},code={}", arch, code));

    build.compile("libvanity.a");

    // Add link directory
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    // Emit the location of the compiled library
    let out_dir = std::env::var("OUT_DIR").unwrap();
    println!("cargo:rustc-link-search=native={}", out_dir);
}
