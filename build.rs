#[cfg(feature = "gpu")]
mod build_native;
#[cfg(feature = "gpu")]
mod build_wsl;

fn main() {
    #[cfg(feature = "gpu")]
    {
        if use_wsl_builder() {
            println!("cargo:warning=Using WSL CUDA builder");
            build_wsl::build_cuda_libs();
        } else {
            println!("cargo:warning=Using native CUDA builder");
            build_native::build_cuda_libs();
        }
    }
}

#[cfg(feature = "gpu")]
fn use_wsl_builder() -> bool {
    use std::{env, fs, path::Path};

    // Manual override always wins:
    // VANITY_BUILD_IMPL=wsl | native
    if let Ok(v) = env::var("VANITY_BUILD_IMPL") {
        let v = v.trim().to_ascii_lowercase();
        if v == "wsl" {
            return true;
        }
        if v == "native" {
            return false;
        }
    }

    // Auto-detect WSL.
    if env::var_os("WSL_DISTRO_NAME").is_some() || env::var_os("WSL_INTEROP").is_some() {
        return true;
    }
    if Path::new("/usr/lib/wsl/lib").exists() {
        return true;
    }
    if let Ok(s) = fs::read_to_string("/proc/version") {
        let sl = s.to_ascii_lowercase();
        if sl.contains("microsoft") || sl.contains("wsl") {
            return true;
        }
    }
    false
}
