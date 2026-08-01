use std::{
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
};

pub fn print_keypair(
    seed: &[u8; 32],
    pubkey: &[u8; 32],
    pubkey_str: &str,
    save: bool,
) -> io::Result<()> {
    let seed_hex: String = seed
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    let mut keypair = [0u8; 64];
    keypair[..32].copy_from_slice(seed);
    keypair[32..].copy_from_slice(pubkey);

    eprintln!("pubkey:   {pubkey_str}");
    eprintln!("seed hex: {seed_hex}");
    eprintln!("keypair json (solana-compatible): {keypair:?}");

    if save {
        let path = PathBuf::from(format!("{pubkey_str}.json"));
        save_keypair(&path, &keypair).map_err(|error| {
            io::Error::new(
                error.kind(),
                format!(
                    "failed to save keypair to {}: {error}",
                    path.display()
                ),
            )
        })?;
        eprintln!("saved keypair: {}", path.display());
    }

    Ok(())
}

fn save_keypair(path: &Path, keypair: &[u8]) -> io::Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);

    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }

    let mut file = options.open(path)?;
    let result =
        writeln!(file, "{keypair:?}").and_then(|_| file.sync_all());
    if result.is_err() {
        drop(file);
        let _ = fs::remove_file(path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_directory() -> PathBuf {
        std::env::temp_dir().join(format!(
            "vanity-keypair-output-{}-{}",
            std::process::id(),
            rand::random::<u64>()
        ))
    }

    #[test]
    fn saves_solana_keypair_without_overwriting() {
        let directory = test_directory();
        fs::create_dir(&directory).unwrap();
        let path = directory.join("example.json");
        let seed = [1u8; 32];
        let pubkey = [2u8; 32];
        let mut keypair = [0u8; 64];
        keypair[..32].copy_from_slice(&seed);
        keypair[32..].copy_from_slice(&pubkey);

        save_keypair(&path, &keypair).unwrap();
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            format!("{keypair:?}\n")
        );

        let error = save_keypair(&path, &[3u8; 64]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            format!("{keypair:?}\n")
        );

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&path)
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }

        fs::remove_dir_all(directory).unwrap();
    }
}
