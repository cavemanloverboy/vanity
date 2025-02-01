use clap::Parser;
use logfather::{Level, Logger};
use num_format::{Locale, ToFormattedString};
use rand::{distributions::Alphanumeric, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};
use solana_pubkey::Pubkey;
use solana_rpc_client::rpc_client::RpcClient;
use solana_sdk::{
    bpf_loader_upgradeable::{self, get_program_data_address, UpgradeableLoaderState},
    instruction::{AccountMeta, Instruction},
    loader_upgradeable_instruction::UpgradeableLoaderInstruction,
    signature::read_keypair_file,
    signer::Signer,
    system_instruction, system_program, sysvar,
    transaction::Transaction,
};

use std::{
    array,
    path::PathBuf,
    str::FromStr,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
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

    /// The target prefix for the pubkey
    #[clap(long)]
    pub prefix: Option<String>,

    /// The target postfix for the pubkey
    #[clap(long)]
    pub postfix: Option<String>,

    /// Whether user cares about the case of the pubkey
    #[clap(long, default_value_t = false)]
    pub case_insensitive: bool,

    /// Optional log file
    #[clap(long)]
    pub logfile: Option<String>,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

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
    let payer_keypair = args
        .payer
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
        &args.seed,
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
        blockhash,
    );

    let sig = rpc_client
        .send_and_confirm_transaction(&transaction)
        .unwrap();
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
    seed: &str,
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
            &bpf_loader_upgradeable::id(),
        ),
        Instruction::new_with_bincode(
            bpf_loader_upgradeable::id(),
            &UpgradeableLoaderInstruction::DeployWithMaxDataLen { max_data_len },
            vec![
                AccountMeta::new(*payer_address, true),
                AccountMeta::new(programdata_address, false),
                AccountMeta::new(*program_address, false),
                AccountMeta::new(*buffer_address, false),
                AccountMeta::new_readonly(sysvar::rent::id(), false),
                AccountMeta::new_readonly(sysvar::clock::id(), false),
                AccountMeta::new_readonly(system_program::id(), false),
                AccountMeta::new_readonly(*upgrade_authority_address, true),
            ],
        ),
    ]
}

fn grind(mut args: GrindArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);

    let prefix = if let Some(prefix) = &args.prefix {
        Some(get_validated_target(prefix, args.case_insensitive))
    } else {
        None
    };

    let postfix = if let Some(postfix) = &args.postfix {
        Some(get_validated_target(postfix, args.case_insensitive))
    } else {
        None
    };

    assert!(
        prefix.is_some() || postfix.is_some(),
        "you must supply a prefix and/or postfix",
    );

    #[cfg(feature = "gpu")]
    assert!(
        prefix.is_some() && postfix.is_none(),
        "gpu grind only supports prefix"
    );

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
    logfather::info!("using {} gpus", args.num_gpus);

    #[cfg(feature = "gpu")]
    let _gpu_threads: Vec<_> = (0..args.num_gpus)
        .map(move |gpu_index| {
            let target = prefix.unwrap();
            std::thread::Builder::new()
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
                            vanity_round(gpu_index, seed.as_ref().as_ptr(), args.base.to_bytes().as_ptr(), args.owner.to_bytes().as_ptr(), target.as_ptr(), target.len() as u64, out.as_mut_ptr(), args.case_insensitive);
                        }
                        let time_sec = timer.elapsed().as_secs_f64();

                        // Reconstruct solution
                        let reconstructed: [u8; 32] = Sha256::new()
                            .chain_update(&args.base)
                            .chain_update(&out[..16])
                            .chain_update(&args.owner)
                            .finalize()
                            .into();
                        let out_str = fd_bs58::encode_32(reconstructed);
                        let out_str_target_check = maybe_bs58_aware_lowercase(&out_str, args.case_insensitive);
                        let count = u64::from_le_bytes(array::from_fn(|i| out[16 + i]));
                        logfather::info!(
                            "{}.. found in {:.3} seconds on gpu {gpu_index:>3}; {:>13} iters; {:>12} iters/sec",
                            &out_str[..(target.len() + 4).min(40)],
                            time_sec,
                            count.to_formatted_string(&Locale::en),
                            ((count as f64 / time_sec) as u64).to_formatted_string(&Locale::en)
                        );

                        if out_str_target_check.starts_with(target) {
                            logfather::info!("out seed = {out:?} -> {}", core::str::from_utf8(&out[..16]).unwrap());
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
            let out_str_target_check = maybe_bs58_aware_lowercase(&pubkey, args.case_insensitive);

            count += 1;

            // Did cpu find prefix?
            let prefix_good = if let Some(prefix) = prefix {
                out_str_target_check.starts_with(prefix)
            } else {
                true
            };

            // Did cpu find postfix?
            let postfix_good = if let Some(postfix) = postfix {
                out_str_target_check.ends_with(postfix)
            } else {
                true
            };

            if prefix_good && postfix_good {
                let time_secs = timer.elapsed().as_secs_f64();
                logfather::info!(
                    "cpu {i} found target: {pubkey}; {seed:?} -> {} in {:.3}s; {} attempts; {} attempts per second",
                    core::str::from_utf8(&seed).unwrap(),
                    time_secs,
                    count.to_formatted_string(&Locale::en),
                    ((count as f64 / time_secs) as u64).to_formatted_string(&Locale::en)
                );

                EXIT.store(true, Ordering::Release);
                break;
            }
        }
    });
}

fn get_validated_target(target: &str, case_insensitive: bool) -> &'static str {
    // Static string of BS58 characters
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Validate target (i.e. does it include 0, O, I, l)
    //
    // maybe TODO: technically we could accept I or o if case-insensitivity but I suspect
    // most users will provide lowercase targets for case-insensitive searches
    for c in target.chars() {
        assert!(
            BS58_CHARS.contains(c),
            "your prefix or postfix contains invalid bs58: {}",
            c
        );
    }

    // bs58-aware lowercase converison
    let target = maybe_bs58_aware_lowercase(target, case_insensitive);

    target.leak()
}

fn maybe_bs58_aware_lowercase(target: &str, case_insensitive: bool) -> String {
    // L is only char that shouldn't be converted to lowercase in case-insensitivity case
    const LOWERCASE_EXCEPTIONS: &str = "L";

    if case_insensitive {
        target
            .chars()
            .map(|c| {
                if LOWERCASE_EXCEPTIONS.contains(c) {
                    c
                } else {
                    c.to_ascii_lowercase()
                }
            })
            .collect::<String>()
    } else {
        target.to_string()
    }
}

extern "C" {
    pub fn vanity_round(
        gpus: u32,
        seed: *const u8,
        base: *const u8,
        owner: *const u8,
        target: *const u8,
        target_len: u64,
        out: *mut u8,
        case_insensitive: bool,
    );
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
