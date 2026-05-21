use std::env;

pub fn build_cuda_libs() {
    fn parse_arches() -> Vec<String> {
        // Keep native default aligned with previous behavior (sm_89).
        let raw = env::var("VANITY_CUDA_ARCH").unwrap_or_else(|_| "89".to_string());
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

    println!("cargo::rerun-if-changed=kernels/");
    println!("cargo:rerun-if-env-changed=VANITY_CUDA_ARCH");

    let mut build = cc::Build::new();
    build
        .cuda(true)
        .include("kernels")
        .file("kernels/utils.cu")
        .file("kernels/vanity.cu")
        .file("kernels/vanity_keypair.cu")
        .file("kernels/base58.cu")
        .file("kernels/sha256.cu")
        .flag("-cudart=static");

    let arches = parse_arches();
    println!("cargo:warning=Native build arch(es): {}", arches.join(","));
    for arch in arches {
        let real = format!("-gencode=arch=compute_{arch},code=sm_{arch}");
        let virtual_ptx = format!("-gencode=arch=compute_{arch},code=compute_{arch}");
        build.flag(&real).flag(&virtual_ptx);
    }

    build.compile("libvanity.a");

    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cuda");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by cargo");
    println!("cargo:rustc-link-search=native={}", out_dir);
}
