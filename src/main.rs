use clap::Parser;
use logfather::{ Level, Logger };
use num_format::{ Locale, ToFormattedString };
use rand::{ distributions::Alphanumeric, Rng };
use rayon::iter::{ IntoParallelIterator, ParallelIterator };
use sha2::{ Digest, Sha256 };
use solana_pubkey::Pubkey;
use solana_rpc_client::rpc_client::RpcClient;
use solana_sdk::{
    bpf_loader_upgradeable::{ self, get_program_data_address, UpgradeableLoaderState },
    instruction::{ AccountMeta, Instruction },
    loader_upgradeable_instruction::UpgradeableLoaderInstruction,
    signature::read_keypair_file,
    signer::Signer,
    system_instruction,
    system_program,
    sysvar,
    transaction::Transaction,
    signature::{ Keypair },
};

use std::{
    array,
    path::PathBuf,
    str::FromStr,
    sync::atomic::{ AtomicBool, Ordering },
    time::Instant,
    fs::{ self, File },
    io::Write,
};

use base64;
use bs58;
use hex;

#[derive(Debug, Parser)]
pub enum Command {
    Grind(GrindArgs),
    Deploy(DeployArgs),
}

#[derive(Debug, Parser)]
pub struct GrindArgs {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: Pubkey,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// The prefix for the pubkey (optional)
    #[clap(long, default_value = "")]
    pub prefix: String,

    /// The suffix for the pubkey (optional)
    #[clap(long, default_value = "")]
    pub suffix: String,

    /// Search for this string anywhere in the address
    #[clap(long, default_value = "")]
    pub any: String,

    /// Whether to ignore case for prefix matching
    #[clap(long = "ci-prefix", default_value_t = false)]
    pub case_insensitive_prefix: bool,

    /// Whether to ignore case for suffix matching
    #[clap(long = "ci-suffix", default_value_t = false)]
    pub case_insensitive_suffix: bool,

    /// Whether to ignore case for 'any' matching
    #[clap(long = "ci-any", default_value_t = false)]
    pub case_insensitive_any: bool,

    /// Whether to match leet speak variants (e.g. a=4, e=3, etc)
    #[clap(long = "leet", default_value_t = false)]
    pub leet_speak: bool,

    /// Optional log file
    #[clap(long)]
    pub logfile: Option<String>,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,
}

#[derive(Debug, Parser)]
pub struct DeployArgs {
    /// The keypair that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long)]
    pub base: PathBuf,

    /// The keypair that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, default_value = "https://api.mainnet-beta.solana.com")]
    pub rpc: String,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// Buffer where the program has been written (via solana program write-buffer)
    #[clap(long, value_parser = parse_pubkey)]
    pub buffer: Pubkey,

    /// Path to keypair that will pay for deploy. when this is None, base is used as payer
    #[clap(long)]
    pub payer: Option<PathBuf>,

    /// Seed grinded via grind
    #[clap(long)]
    pub seed: String,

    /// Program authority (default is (payer) keypair's pubkey)
    #[clap(long)]
    pub authority: Option<Pubkey>,

    /// Compute unit price
    #[clap(long)]
    pub compute_unit_price: Option<u64>,

    /// Optional log file
    #[clap(long)]
    pub logfile: Option<String>,
}

static EXIT: AtomicBool = AtomicBool::new(false);

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    // Parse command line arguments
    let command = Command::parse();
    match command {
        Command::Grind(args) => {
            grind(args);
        }

        Command::Deploy(args) => {
            deploy(args);
        }
    }
}

fn deploy(args: DeployArgs) {
    // Load base and payer keypair
    let base_keypair = read_keypair_file(&args.base).expect("failed to read base keypair");
    let payer_keypair = args.payer
        .as_ref()
        .map(|payer| read_keypair_file(payer).expect("failed to read payer keypair"))
        .unwrap_or(base_keypair.insecure_clone());
    let authority = args.authority.unwrap_or_else(|| payer_keypair.pubkey());

    // Target
    let target = Pubkey::create_with_seed(&base_keypair.pubkey(), &args.seed, &args.owner).unwrap();

    // Fetch rent
    let rpc_client = RpcClient::new(args.rpc);
    // this is such a dumb way to do this
    let buffer_len = rpc_client.get_account_data(&args.buffer).unwrap().len();
    // I forgot the header len so let's just add 64 for now lol
    let rent = rpc_client
        .get_minimum_balance_for_rent_exemption(UpgradeableLoaderState::size_of_program())
        .expect("failed to fetch rent");

    // Create account with seed
    let instructions = deploy_with_max_program_len_with_seed(
        &payer_keypair.pubkey(),
        &target,
        &args.buffer,
        &authority,
        rent,
        64 + buffer_len,
        &base_keypair.pubkey(),
        &args.seed
    );
    // Transaction
    let blockhash = rpc_client.get_latest_blockhash().unwrap();
    let signers = if args.payer.is_none() {
        vec![&base_keypair]
    } else {
        vec![&base_keypair, &payer_keypair]
    };
    let transaction = Transaction::new_signed_with_payer(
        &instructions,
        Some(&payer_keypair.pubkey()),
        &signers,
        blockhash
    );

    let sig = rpc_client.send_and_confirm_transaction(&transaction).unwrap();
    println!("Deployed {target}: {sig}");
}

pub fn deploy_with_max_program_len_with_seed(
    payer_address: &Pubkey,
    program_address: &Pubkey,
    buffer_address: &Pubkey,
    upgrade_authority_address: &Pubkey,
    program_lamports: u64,
    max_data_len: usize,
    base: &Pubkey,
    seed: &str
) -> [Instruction; 2] {
    let programdata_address = get_program_data_address(program_address);
    [
        system_instruction::create_account_with_seed(
            payer_address,
            program_address,
            base,
            seed,
            program_lamports,
            UpgradeableLoaderState::size_of_program() as u64,
            &bpf_loader_upgradeable::id()
        ),
        Instruction::new_with_bincode(
            bpf_loader_upgradeable::id(),
            &(UpgradeableLoaderInstruction::DeployWithMaxDataLen { max_data_len }),
            vec![
                AccountMeta::new(*payer_address, true),
                AccountMeta::new(programdata_address, false),
                AccountMeta::new(*program_address, false),
                AccountMeta::new(*buffer_address, false),
                AccountMeta::new_readonly(sysvar::rent::id(), false),
                AccountMeta::new_readonly(sysvar::clock::id(), false),
                AccountMeta::new_readonly(system_program::id(), false),
                AccountMeta::new_readonly(*upgrade_authority_address, true)
            ]
        ),
    ]
}

fn grind(mut args: GrindArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);

    let (prefix, suffix, any) = get_validated_strings(&args);

    // Initialize logger with optional logfile
    let mut logger = Logger::new();
    if let Some(ref logfile) = args.logfile {
        logger.file(true);
        logger.path(logfile);
    }

    // Slightly more compact log format
    logger.log_format("[{timestamp} {level}] {message}");
    logger.timestamp_format("%Y-%m-%d %H:%M:%S");
    logger.level(Level::Info);

    // Print resource usage
    logfather::info!("using {} threads", args.num_cpus);

    #[cfg(feature = "gpu")]
    let num_gpus = unsafe { get_gpu_count() };
    #[cfg(feature = "gpu")]
    logfather::info!("detected {} GPUs", num_gpus);

    #[cfg(feature = "gpu")]
    let _gpu_threads: Vec<_> = (0..num_gpus)
        .map(move |gpu_index| {
            std::thread::Builder
                ::new()
                .name(format!("gpu{gpu_index}"))
                .spawn(move || {
                    logfather::trace!("starting gpu {gpu_index}");

                    // Pre-allocate buffers
                    let mut out = [0u8; 24];
                    let mut pubkey_bytes = [0u8; 32];
                    let mut hasher = Sha256::new();

                    for iteration in 0_u64.. {
                        if EXIT.load(Ordering::SeqCst) {
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        }

                        // Generate new seed for this gpu & iteration
                        let seed = new_gpu_seed(gpu_index, iteration);
                        let timer = Instant::now();

                        unsafe {
                            vanity_round(
                                gpu_index,
                                seed.as_ref().as_ptr(),
                                args.base.to_bytes().as_ptr(),
                                args.owner.to_bytes().as_ptr(),
                                prefix.as_ptr(),
                                prefix.len() as u64,
                                suffix.as_ptr(),
                                suffix.len() as u64,
                                any.as_ptr(),
                                any.len() as u64,
                                out.as_mut_ptr(),
                                args.case_insensitive_prefix,
                                args.case_insensitive_suffix,
                                args.case_insensitive_any,
                                args.leet_speak
                            );
                        }

                        let time_sec = timer.elapsed().as_secs_f64();

                        // Reuse hasher and buffer for pubkey calculation
                        hasher.reset();
                        hasher.update(&args.base);
                        hasher.update(&out[..16]);
                        hasher.update(&args.owner);
                        pubkey_bytes.copy_from_slice(&hasher.finalize_reset());

                        let pubkey = fd_bs58::encode_32(pubkey_bytes);
                        let count = u64::from_le_bytes(array::from_fn(|i| out[16 + i]));

                        logfather::info!(
                            "{}.. found in {:.3} seconds on gpu {gpu_index:>3}; {:>13} iters; {:>12} iters/sec",
                            &pubkey[..(prefix.len() + suffix.len() + 4).min(40)],
                            time_sec,
                            count.to_formatted_string(&Locale::en),
                            (((count as f64) / time_sec) as u64).to_formatted_string(&Locale::en)
                        );

                        let rust_matches = matches_vanity_key(
                            &pubkey,
                            prefix,
                            suffix,
                            any,
                            args.case_insensitive_prefix,
                            args.case_insensitive_suffix,
                            args.case_insensitive_any,
                            args.leet_speak
                        );

                        if !rust_matches && (count > 0 || out[16..24].iter().any(|&x| x != 0)) {
                            logfather::error!("\nMISMATCH DETECTED!");
                            logfather::error!("CUDA found a match but Rust validation failed");
                            logfather::error!("Address: {}", pubkey);
                            logfather::error!("Search criteria:");
                            logfather::error!("  Prefix: '{}'", prefix);
                            logfather::error!("  Suffix: '{}'", suffix);
                            logfather::error!("  Any: '{}'", any);
                            logfather::error!("  Case insensitive: {}", if
                                args.case_insensitive_prefix
                            {
                                "prefix enabled"
                            } else {
                                "prefix disabled"
                            });
                            logfather::error!("  Case insensitive: {}", if
                                args.case_insensitive_suffix
                            {
                                "suffix enabled"
                            } else {
                                "suffix disabled"
                            });
                            logfather::error!("  Case insensitive: {}", if
                                args.case_insensitive_any
                            {
                                "any enabled"
                            } else {
                                "any disabled"
                            });
                            logfather::error!("  Leet speak: {}", if args.leet_speak {
                                "enabled"
                            } else {
                                "disabled"
                            });
                            logfather::error!("Seed info:");
                            logfather::error!("  Bytes: {:?}", &out[..16]);
                            logfather::error!(
                                "  UTF-8: {}",
                                core::str::from_utf8(&out[..16]).unwrap_or("Invalid UTF-8")
                            );
                        }

                        if rust_matches {
                            logfather::info!("\nGPU MATCH FOUND!");
                            logfather::info!("Full address: {}", pubkey);
                            print_key_details(&out[..16], &args.base, &args.owner);

                            if let Err(err) = save_vanity_key(&pubkey, &out[..16]) {
                                logfather::error!("{}", err);
                                return;
                            }
                            EXIT.store(true, Ordering::SeqCst);
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        }
                    }
                })
                .unwrap()
        })
        .collect();

    (0..args.num_cpus).into_par_iter().for_each(|i| {
        let timer = Instant::now();
        let mut count = 0_u64;

        let base_sha = Sha256::new().chain_update(args.base);
        loop {
            if EXIT.load(Ordering::Acquire) {
                return;
            }

            let mut seed_iter = rand::thread_rng().sample_iter(&Alphanumeric).take(16);
            let seed: [u8; 16] = array::from_fn(|_| seed_iter.next().unwrap());

            let pubkey_bytes: [u8; 32] = base_sha
                .clone()
                .chain_update(seed)
                .chain_update(args.owner)
                .finalize()
                .into();
            let pubkey = fd_bs58::encode_32(pubkey_bytes);

            count += 1;

            if
                matches_vanity_key(
                    &pubkey,
                    prefix,
                    suffix,
                    any,
                    args.case_insensitive_prefix,
                    args.case_insensitive_suffix,
                    args.case_insensitive_any,
                    args.leet_speak
                )
            {
                let time_secs = timer.elapsed().as_secs_f64();
                logfather::info!(
                    "cpu {i} found key: {pubkey}; {seed:?} -> {} in {:.3}s; {} attempts; {} attempts per second",
                    core::str::from_utf8(&seed).unwrap(),
                    time_secs,
                    count.to_formatted_string(&Locale::en),
                    (((count as f64) / time_secs) as u64).to_formatted_string(&Locale::en)
                );

                print_key_details(&seed, &args.base, &args.owner);

                if let Err(err) = save_vanity_key(&pubkey, &seed) {
                    logfather::error!("{}", err);
                    return;
                }
                EXIT.store(true, Ordering::SeqCst);
                return;
            }
        }
    });
}

fn get_validated_strings(args: &GrindArgs) -> (&'static str, &'static str, &'static str) {
    // Static string of BS58 characters
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Map of invalid chars to their BS58 equivalents
    const CHAR_MAP: &[(char, char)] = &[
        ('0', 'o'), // Map 0 to o
        ('I', '1'), // Map I to 1
        ('O', 'o'), // Map O to o
        ('l', 'L'), // Map l to L
    ];

    // Helper to convert invalid chars to valid BS58
    fn convert_to_valid_bs58(c: char) -> char {
        if BS58_CHARS.contains(c) {
            return c;
        }

        // Try to find a valid replacement
        for (invalid, valid) in CHAR_MAP {
            if c == *invalid {
                return *valid;
            }
        }

        // No valid replacement found
        panic!("Character '{}' cannot be converted to a valid base58 character", c);
    }

    // Convert strings and validate
    let prefix: String = args.prefix.chars().map(convert_to_valid_bs58).collect();
    let suffix: String = args.suffix.chars().map(convert_to_valid_bs58).collect();
    let any: String = args.any.chars().map(convert_to_valid_bs58).collect();

    // bs58-aware lowercase conversion for all strings
    let prefix = maybe_bs58_aware_lowercase(&prefix, args.case_insensitive_prefix);
    let suffix = maybe_bs58_aware_lowercase(&suffix, args.case_insensitive_suffix);
    let any = maybe_bs58_aware_lowercase(&any, args.case_insensitive_any);

    (prefix.leak(), suffix.leak(), any.leak())
}

fn maybe_bs58_aware_lowercase(item: &str, case_insensitive: bool) -> String {
    // L is only char that shouldn't be converted to lowercase in case-insensitivity case
    const LOWERCASE_EXCEPTIONS: &str = "L";

    if case_insensitive {
        item.chars()
            .map(|c| {
                if LOWERCASE_EXCEPTIONS.contains(c) { c } else { c.to_ascii_lowercase() }
            })
            .collect::<String>()
    } else {
        item.to_string()
    }
}

#[cfg(feature = "gpu")]
extern "C" {
    pub fn vanity_round(
        gpus: i32,
        seed: *const u8,
        base: *const u8,
        owner: *const u8,
        prefix: *const u8,
        prefix_len: u64,
        suffix: *const u8,
        suffix_len: u64,
        any: *const u8,
        any_len: u64,
        out: *mut u8,
        case_insensitive_prefix: bool,
        case_insensitive_suffix: bool,
        case_insensitive_any: bool,
        leet_speak: bool
    );

    fn get_gpu_count() -> i32;
}

#[cfg(feature = "gpu")]
fn new_gpu_seed(gpu_id: i32, iteration: u64) -> [u8; 32] {
    Sha256::new()
        .chain_update(rand::random::<[u8; 32]>())
        .chain_update(gpu_id.to_le_bytes())
        .chain_update(iteration.to_le_bytes())
        .finalize()
        .into()
}

fn parse_pubkey(input: &str) -> Result<Pubkey, String> {
    Pubkey::from_str(input).map_err(|e| e.to_string())
}

fn maybe_update_num_cpus(num_cpus: &mut u32) {
    if *num_cpus == 0 {
        *num_cpus = rayon::current_num_threads() as u32;
    }
}

fn print_key_details(seed: &[u8], base: &Pubkey, owner: &Pubkey) {
    // 1. Print raw seed in different formats
    println!("\nSeed Details:");
    println!("Raw bytes: {:?}", seed);
    println!("Hex: {}", hex::encode(seed));
    println!("Base64: {}", base64::engine::general_purpose::STANDARD.encode(seed));
    println!("UTF-8: {}", String::from_utf8_lossy(seed));

    // 2. Calculate and print derived address
    let mut hasher = Sha256::new();
    hasher.update(base.to_bytes());
    hasher.update(seed);
    hasher.update(owner.to_bytes());
    let pubkey_bytes: [u8; 32] = hasher.finalize().into();

    // Print address in different formats
    println!("\nDerived Address Details:");
    println!("Raw bytes: {:?}", pubkey_bytes);
    println!("Hex: {}", hex::encode(&pubkey_bytes));
    println!("Base58: {}", bs58::encode(&pubkey_bytes).into_string());
    println!("Base64: {}", base64::engine::general_purpose::STANDARD.encode(&pubkey_bytes));

    // 3. Try to interpret as Solana keypair
    if seed.len() >= 32 {
        if let Ok(keypair) = Keypair::from_bytes(&seed[..32]) {
            println!("\nInterpreted as Solana Keypair:");
            println!("Public Key: {}", keypair.pubkey());
            println!("Base58 Secret Key: {}", bs58::encode(keypair.secret()).into_string());
            println!("Hex Secret Key: {}", hex::encode(keypair.secret()));
            println!(
                "Base64 Secret Key: {}",
                base64::engine::general_purpose::STANDARD.encode(keypair.secret())
            );
        }
    }

    // 4. Print validation info
    println!("\nValidation:");
    let encoded = fd_bs58::encode_32(pubkey_bytes);
    println!("fd_bs58 encoded (44): {}", encoded);
    println!("bs58 encoded: {}", bs58::encode(&pubkey_bytes).into_string());
    if encoded != bs58::encode(&pubkey_bytes).into_string() {
        println!("WARNING: Encoding mismatch between fd_bs58 and bs58!");
    }
}

fn save_vanity_key(pubkey: &str, seed: &[u8]) -> Result<(), String> {
    let output_dir = PathBuf::from("keys");
    logfather::debug!("Checking output directory: {}", output_dir.display());

    // Check if directory exists first
    if !output_dir.exists() {
        logfather::debug!("Output directory does not exist, creating it");
        if let Err(err) = fs::create_dir_all(&output_dir) {
            return Err(format!("Failed to create output directory: {}", err));
        }
        logfather::info!("Created output directory: {}", output_dir.display());
    } else {
        logfather::debug!("Output directory already exists");
    }

    let output_file_path = output_dir.join(format!("{}.txt", pubkey));
    logfather::debug!("Generated output file path: {}", output_file_path.display());

    let mut file = File::create(&output_file_path).map_err(|err|
        format!("Error opening file {}: {}", output_file_path.display(), err)
    )?;

    // Write original format
    writeln!(
        file,
        "{} -> {:?} [{}]",
        pubkey,
        seed,
        core::str::from_utf8(seed).unwrap_or("Invalid UTF-8")
    ).map_err(|err| format!("Error writing to file {}: {}", output_file_path.display(), err))?;

    // Write additional formats
    writeln!(file, "\nSeed formats:").unwrap();
    writeln!(file, "Hex: {}", hex::encode(seed)).unwrap();
    writeln!(file, "Base64: {}", base64::engine::general_purpose::STANDARD.encode(seed)).unwrap();

    logfather::info!("Successfully saved output to {}", output_file_path.display());
    Ok(())
}

fn generate_leet_patterns(input: &str) -> Vec<String> {
    if input.is_empty() {
        return vec![String::new()];
    }

    // Pre-allocate with a reasonable capacity
    let mut patterns = Vec::with_capacity(input.len() * 2);
    patterns.push(input.to_string());

    // First pass: generate patterns by replacing letters with numbers
    for i in 0..input.len() {
        let c = input.chars().nth(i).unwrap();
        let replacement = match c {
            'a' | 'A' => Some("4"),
            'e' | 'E' => Some("3"),
            't' | 'T' => Some("7"),
            'l' | 'L' | 'i' | 'I' => Some("1"),
            's' | 'S' => Some("5"),
            'g' | 'G' => Some("6"),
            'b' | 'B' => Some("8"),
            'z' | 'Z' => Some("2"),
            _ => None,
        };

        if let Some(repl) = replacement {
            let mut new_pattern = String::with_capacity(input.len());
            new_pattern.push_str(&input[..i]);
            new_pattern.push_str(repl);
            new_pattern.push_str(&input[i + 1..]);
            patterns.push(new_pattern);
        }
    }

    // Second pass: generate patterns by replacing numbers with letters
    let mut number_patterns = Vec::with_capacity(patterns.len() * 2);
    for pattern in &patterns {
        for i in 0..pattern.len() {
            let c = pattern.chars().nth(i).unwrap();
            let replacements = match c {
                '4' => vec!["a", "A"],
                '3' => vec!["e", "E"],
                '7' => vec!["t", "T"],
                '1' => vec!["l", "L", "i", "I"],
                '5' => vec!["s", "S"],
                '6' => vec!["g", "G"],
                '8' => vec!["b", "B"],
                '2' => vec!["z", "Z"],
                _ => vec![],
            };

            for repl in replacements {
                let mut new_pattern = String::with_capacity(pattern.len());
                new_pattern.push_str(&pattern[..i]);
                new_pattern.push_str(repl);
                new_pattern.push_str(&pattern[i + 1..]);
                number_patterns.push(new_pattern);
            }
        }
    }

    patterns.extend(number_patterns);
    patterns.sort_unstable(); // Use sort_unstable for better performance
    patterns.dedup();
    patterns
}

fn get_search_patterns(
    prefix: &str,
    suffix: &str,
    any: &str,
    leet_speak: bool
) -> (Vec<String>, Vec<String>, Vec<String>) {
    if leet_speak {
        let prefix_patterns = generate_leet_patterns(prefix);
        let suffix_patterns = generate_leet_patterns(suffix);
        let any_patterns = generate_leet_patterns(any);

        logfather::debug!("Generated leet patterns:");
        logfather::debug!("  Prefix patterns: {:?}", prefix_patterns);
        logfather::debug!("  Suffix patterns: {:?}", suffix_patterns);
        logfather::debug!("  Any patterns: {:?}", any_patterns);

        (prefix_patterns, suffix_patterns, any_patterns)
    } else {
        (vec![prefix.to_string()], vec![suffix.to_string()], vec![any.to_string()])
    }
}

fn check_matches(check_str: &str, patterns: &[String], match_type: &str) -> bool {
    let matches = match match_type {
        "prefix" =>
            patterns.iter().any(|p| {
                if p.is_empty() {
                    return true;
                }
                if check_str.starts_with(p) {
                    return true;
                }
                // Only generate patterns if needed
                if p.len() <= check_str.len() {
                    let substr = &check_str[..p.len()];
                    let address_patterns = generate_leet_patterns(substr);
                    address_patterns.iter().any(|ap| ap == p)
                } else {
                    false
                }
            }),
        "suffix" =>
            patterns.iter().any(|s| {
                if s.is_empty() {
                    return true;
                }
                if check_str.ends_with(s) {
                    return true;
                }
                // Only generate patterns if needed
                if s.len() <= check_str.len() {
                    let substr = &check_str[check_str.len() - s.len()..];
                    let address_patterns = generate_leet_patterns(substr);
                    address_patterns.iter().any(|ap| ap == s)
                } else {
                    false
                }
            }),
        "any" =>
            patterns.iter().any(|a| {
                if a.is_empty() {
                    return true;
                }
                if check_str.contains(a) {
                    return true;
                }
                // Only generate patterns for substrings of matching length
                if a.len() <= check_str.len() {
                    for i in 0..=check_str.len() - a.len() {
                        let substr = &check_str[i..i + a.len()];
                        let address_patterns = generate_leet_patterns(substr);
                        if address_patterns.iter().any(|ap| ap == a) {
                            return true;
                        }
                    }
                }
                false
            }),
        _ => false,
    };
    logfather::debug!("  {} match: {}", match_type, matches);
    matches
}

fn matches_vanity_key(
    pubkey_str: &str,
    prefix: &str,
    suffix: &str,
    any: &str,
    case_insensitive_prefix: bool,
    case_insensitive_suffix: bool,
    case_insensitive_any: bool,
    leet_speak: bool
) -> bool {
    logfather::debug!("\nRust checking address: {}", pubkey_str);
    logfather::debug!("Search criteria:");
    logfather::debug!("  Prefix: '{}' (len={})", prefix, prefix.len());
    logfather::debug!("  Suffix: '{}' (len={})", suffix, suffix.len());
    logfather::debug!("  Any: '{}' (len={})", any, any.len());
    logfather::debug!("  Case insensitive: {}", if case_insensitive_prefix {
        "prefix enabled"
    } else {
        "prefix disabled"
    });
    logfather::debug!("  Case insensitive: {}", if case_insensitive_suffix {
        "suffix enabled"
    } else {
        "suffix disabled"
    });
    logfather::debug!("  Case insensitive: {}", if case_insensitive_any {
        "any enabled"
    } else {
        "any disabled"
    });
    logfather::debug!("  Leet speak: {}", if leet_speak { "enabled" } else { "disabled" });

    let check_str = if case_insensitive_prefix {
        maybe_bs58_aware_lowercase(pubkey_str, true)
    } else {
        pubkey_str.to_string()
    };
    logfather::debug!("After case conversion: {}", check_str);

    let (prefix_patterns, suffix_patterns, any_patterns) = get_search_patterns(
        prefix,
        suffix,
        any,
        leet_speak
    );

    let prefix_matches = check_matches(&check_str, &prefix_patterns, "prefix");
    let suffix_matches = check_matches(&check_str, &suffix_patterns, "suffix");
    let any_matches = check_matches(&check_str, &any_patterns, "any");

    prefix_matches && suffix_matches && any_matches
}
