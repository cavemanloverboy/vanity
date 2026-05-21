use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
};

pub fn build_cuda_libs() {
    fn resolve_nvcc() -> String {
        if let Ok(nvcc) = env::var("NVCC") {
            if Path::new(&nvcc).exists() {
                return nvcc;
            }
        }
        for candidate in ["/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"] {
            if Path::new(candidate).exists() {
                return candidate.to_string();
            }
        }
        panic!(
            "Could not find nvcc. Install CUDA toolkit in WSL and ensure nvcc is in PATH, \
or set NVCC explicitly (e.g. NVCC=/usr/local/cuda/bin/nvcc)."
        );
    }

    fn parse_arches() -> Vec<String> {
        // Default to RTX 3070 (GA104) = compute capability 8.6 (sm_86).
        // Override via VANITY_CUDA_ARCH, e.g. "89" or "86,89".
        let raw = env::var("VANITY_CUDA_ARCH").unwrap_or_else(|_| "86".to_string());
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| {
                s.trim_start_matches("sm_")
                    .trim_start_matches("compute_")
                    .to_string()
            })
            .collect()
    }

    fn run_checked(cmd: &mut Command, context: &str) {
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("{context}: failed to spawn command: {e}"));
        if !status.success() {
            panic!("{context}: command exited with status {status}");
        }
    }

    let sources = [
        "kernels/utils.cu",
        "kernels/vanity.cu",
        "kernels/vanity_keypair.cu",
        "kernels/base58.cu",
        "kernels/sha256.cu",
    ];
    for src in sources {
        println!("cargo:rerun-if-changed={src}");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=VANITY_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDAHOSTCXX");

    let nvcc = resolve_nvcc();
    let arches = parse_arches();
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set by cargo"));
    let lib_path = out_dir.join("libvanity.a");

    println!("cargo:warning=Using nvcc at {}", nvcc);
    println!("cargo:warning=Target CUDA arch(es): {}", arches.join(","));
    println!("cargo:warning=Compiling CUDA via direct nvcc compile+dlink path");
    println!("cargo:warning=BUILD_RS_REV=2026-04-14-explicit-dlink-stdcpp");

    let mut objects = Vec::with_capacity(sources.len());
    for src in sources {
        let stem = Path::new(src)
            .file_stem()
            .expect("source file should have stem")
            .to_string_lossy()
            .into_owned();
        let obj = out_dir.join(format!("{stem}.o"));
        objects.push(obj.clone());

        let mut cmd = Command::new(&nvcc);
        cmd.arg("-c")
            .arg("-o")
            .arg(&obj)
            .arg("-O3")
            .arg("-Xcompiler=-fPIC")
            .arg("-cudart=shared")
            .arg("-rdc=true")
            .arg("-I")
            .arg("kernels");

        if let Ok(host_cxx) = env::var("CUDAHOSTCXX") {
            if !host_cxx.trim().is_empty() {
                cmd.arg("-ccbin").arg(host_cxx);
            }
        }

        for arch in &arches {
            cmd.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
            cmd.arg(format!("-gencode=arch=compute_{arch},code=compute_{arch}"));
        }
        cmd.arg(src);
        run_checked(&mut cmd, &format!("nvcc -c {src}"));
    }

    let dlink_obj = out_dir.join("vanity_dlink.o");
    let mut dlink = Command::new(&nvcc);
    dlink
        .arg("-dlink")
        .arg("-o")
        .arg(&dlink_obj)
        .arg("-cudart=shared");
    if let Ok(host_cxx) = env::var("CUDAHOSTCXX") {
        if !host_cxx.trim().is_empty() {
            dlink.arg("-ccbin").arg(host_cxx);
        }
    }
    for arch in &arches {
        dlink.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
        dlink.arg(format!("-gencode=arch=compute_{arch},code=compute_{arch}"));
    }
    for obj in &objects {
        dlink.arg(obj);
    }
    dlink.arg("-lcudadevrt");
    run_checked(&mut dlink, "nvcc -dlink");

    let mut ar = Command::new("ar");
    ar.arg("crs").arg(&lib_path);
    for obj in &objects {
        ar.arg(obj);
    }
    ar.arg(&dlink_obj);
    run_checked(&mut ar, "ar crs libvanity.a");

    let mut ranlib = Command::new("ranlib");
    ranlib.arg(&lib_path);
    run_checked(&mut ranlib, "ranlib libvanity.a");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=vanity");

    let lib_search_dirs = [
        "/usr/local/cuda/lib64",
        "/usr/lib/wsl/lib",
        "/usr/lib/x86_64-linux-gnu",
    ];
    for dir in lib_search_dirs {
        if Path::new(dir).exists() {
            println!("cargo:rustc-link-search=native={dir}");
        }
    }

    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cudadevrt");
    println!("cargo:rustc-link-lib=stdc++");
    // Don't link libcuda directly. cudart loads the driver at runtime, and
    // many WSL installs expose only libcuda.so.1 without a libcuda.so dev symlink.
    println!("cargo:warning=Skipping direct -lcuda link (WSL-friendly)");
}
