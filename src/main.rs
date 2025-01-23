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

    /// Whether user cares about the case of the pubkey
    #[clap(long, default_value_t = false)]
    pub case_insensitive: bool,

    /// Whether to match leet speak variants (e.g. a=4, e=3, etc)
    #[clap(long, default_value_t = false)]
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

                    let mut out = [0; 24];
                    for iteration in 0_u64.. {
                        // Exit if a thread found a solution
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
                                args.case_insensitive,
                                args.leet_speak
                            );
                        }
                        let time_sec = timer.elapsed().as_secs_f64();

                        // Reconstruct solution
                        let pubkey_bytes: [u8; 32] = Sha256::new()
                            .chain_update(&args.base)
                            .chain_update(&out[..16])
                            .chain_update(&args.owner)
                            .finalize()
                            .into();
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
                            args.case_insensitive,
                            args.leet_speak
                        );

                        // If CUDA found a match but Rust validation fails, print debug info
                        if !rust_matches && (count > 0 || out[16..24].iter().any(|&x| x != 0)) {
                            logfather::error!("\nMISMATCH DETECTED!");
                            logfather::error!("CUDA found a match but Rust validation failed");
                            logfather::error!("Address: {}", pubkey);
                            logfather::error!("Search criteria:");
                            logfather::error!("  Prefix: '{}'", prefix);
                            logfather::error!("  Suffix: '{}'", suffix);
                            logfather::error!("  Any: '{}'", any);
                            logfather::error!("  Case insensitive: {}", if args.case_insensitive {
                                "enabled"
                            } else {
                                "disabled"
                            });
                            logfather::error!("  Leet speak: {}", if args.leet_speak {
                                "enabled"
                            } else {
                                "disabled"
                            });
                        }

                        if rust_matches {
                            logfather::info!("\nGPU MATCH FOUND!");
                            logfather::info!("Full address: {}", pubkey);
                            logfather::info!("Seed: {:?}", &out[..16]);
                            logfather::info!(
                                "UTF-8 seed: {}",
                                core::str::from_utf8(&out[..16]).unwrap_or("Invalid UTF-8")
                            );

                            let output_dir = PathBuf::from("/mnt/f/coding/vanity/keys");
                            if let Err(err) = save_vanity_key(&pubkey, &out[..16], &output_dir) {
                                logfather::error!("{}", err);
                                return;
                            }
                            EXIT.store(true, Ordering::SeqCst);
                            logfather::trace!("gpu thread {gpu_index} exiting");
                            return;
                        } else {
                            logfather::debug!("pubkey does not match prefix or suffix");
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
                    args.case_insensitive,
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

                let output_dir = PathBuf::from("/mnt/f/coding/vanity/keys");
                if let Err(err) = save_vanity_key(&pubkey, &seed, &output_dir) {
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
        ('l', '1'), // Map l to 1
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
    let prefix = maybe_bs58_aware_lowercase(&prefix, args.case_insensitive);
    let suffix = maybe_bs58_aware_lowercase(&suffix, args.case_insensitive);
    let any = maybe_bs58_aware_lowercase(&any, args.case_insensitive);

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

extern "C" {
    pub fn vanity_round(
        gpus: u32,
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
        case_insensitive: bool,
        leet_speak: bool
    );

    fn get_gpu_count() -> i32;
}

#[cfg(feature = "gpu")]
fn new_gpu_seed(gpu_id: u32, iteration: u64) -> [u8; 32] {
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

fn save_vanity_key(pubkey: &str, seed: &[u8], output_dir: &PathBuf) -> Result<(), String> {
    logfather::debug!("Ensuring output directory exists: {}", output_dir.display());

    if let Err(err) = fs::create_dir_all(output_dir) {
        return Err(format!("Failed to create output directory: {}", err));
    }

    let output_file_path = output_dir.join(format!("{}.txt", pubkey));
    logfather::debug!("Generated output file path: {}", output_file_path.display());

    let mut file = File::create(&output_file_path).map_err(|err|
        format!("Error opening file {}: {}", output_file_path.display(), err)
    )?;

    logfather::debug!("Opened file for writing: {}", output_file_path.display());

    write!(
        file,
        "{} -> {:?} [{}]",
        pubkey,
        seed,
        core::str::from_utf8(seed).unwrap_or("Invalid UTF-8")
    ).map_err(|err| format!("Error writing to file {}: {}", output_file_path.display(), err))?;

    logfather::info!("Successfully saved output to {}", output_file_path.display());
    Ok(())
}

fn leet_transform(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        // Try both letter-to-number and number-to-letter transformations
        let transformed = match c {
            // Letter to number transformations
            'z' | 'Z' => '2',
            'e' | 'E' => '3',
            'a' | 'A' => '4',
            's' | 'S' => '5',
            'g' | 'G' => '6',
            't' | 'T' => '7',
            'b' | 'B' => '8',
            'i' | 'I' | 'l' | 'L' => '1',
            // Number to letter transformations
            '2' => 'z',
            '3' => 'e',
            '4' => 'a',
            '5' => 's',
            '6' => 'g',
            '7' => 't',
            '8' => 'b',
            '1' => 'i',
            // Keep other characters unchanged
            _ => c,
        };
        result.push(transformed);
    }
    result
}

fn matches_vanity_key(
    pubkey_str: &str,
    prefix: &str,
    suffix: &str,
    any: &str,
    case_insensitive: bool,
    leet_speak: bool
) -> bool {
    logfather::debug!("\nRust checking address: {}", pubkey_str);
    logfather::debug!("Search criteria:");
    logfather::debug!("  Prefix: '{}' (len={})", prefix, prefix.len());
    logfather::debug!("  Suffix: '{}' (len={})", suffix, suffix.len());
    logfather::debug!("  Any: '{}' (len={})", any, any.len());
    logfather::debug!("  Case insensitive: {}", if case_insensitive {
        "enabled"
    } else {
        "disabled"
    });
    logfather::debug!("  Leet speak: {}", if leet_speak { "enabled" } else { "disabled" });

    let check_str = maybe_bs58_aware_lowercase(pubkey_str, case_insensitive);
    logfather::debug!("After case conversion: {}", check_str);

    // Transform both the address and search terms for comparison
    let (check_str, prefix, suffix, any) = if leet_speak {
        let transformed_addr = leet_transform(&check_str);
        logfather::debug!("After leet transform of address: {}", transformed_addr);

        // Also transform search terms to match all possible combinations
        let transformed_prefix = leet_transform(prefix);
        let transformed_suffix = leet_transform(suffix);
        let transformed_any = leet_transform(any);

        logfather::debug!("After leet transform of search terms:");
        logfather::debug!("  Prefix: '{}'", transformed_prefix);
        logfather::debug!("  Suffix: '{}'", transformed_suffix);
        logfather::debug!("  Any: '{}'", transformed_any);

        (transformed_addr, transformed_prefix, transformed_suffix, transformed_any)
    } else {
        (check_str, prefix.to_string(), suffix.to_string(), any.to_string())
    };

    let prefix_matches = prefix.is_empty() || check_str.starts_with(&prefix);
    let suffix_matches = suffix.is_empty() || check_str.ends_with(&suffix);
    let any_matches = any.is_empty() || check_str.contains(&any);

    logfather::debug!("Match results:");
    logfather::debug!("  Prefix match: {}", prefix_matches);
    logfather::debug!("  Suffix match: {}", suffix_matches);
    logfather::debug!("  Any match: {}", any_matches);
    logfather::debug!("  Case insensitive: {}", if case_insensitive {
        "enabled"
    } else {
        "disabled"
    });
    logfather::debug!("  Leet speak: {}", if leet_speak { "enabled" } else { "disabled" });

    prefix_matches && suffix_matches && any_matches
}
