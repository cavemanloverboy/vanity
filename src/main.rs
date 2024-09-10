use clap::Parser;
use logfather::{Level, Logger};
use num_format::{Locale, ToFormattedString};
use rand::{distributions::Alphanumeric, Rng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};

use core::str;
use std::{
    array,
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

#[derive(Debug, Parser)]
pub struct Args {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: [u8; 32],

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: [u8; 32],

    /// The target prefix for the pubkey
    #[clap(long)]
    pub target: String,

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

static EXIT: AtomicBool = AtomicBool::new(false);

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    // Parse command line arguments
    let mut args = Args::parse();
    maybe_update_num_cpus(&mut args.num_cpus);

    let target = get_validated_target(&args);

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
                            vanity_round(gpu_index, seed.as_ref().as_ptr(), args.base.as_ptr(), args.owner.as_ptr(), target.as_ptr(), target.len() as u64, out.as_mut_ptr(), args.case_insensitive);
                        }
                        let time_sec = timer.elapsed().as_secs_f64();

                        // Reconstruct solution
                        let reconstructed: [u8; 32] = Sha256::new()
                            .chain_update(&args.base)
                            .chain_update(&out[..16])
                            .chain_update(&args.owner)
                            .finalize()
                            .into();
                        let mut encoded = [0u8; five8::BASE58_ENCODED_32_MAX_LEN];
                        let len = five8::encode_32(reconstructed, &mut encoded);
                        let out_str = str::from_utf8(encoded[..len as usize]).unwrap();
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
                            logfather::info!("out seed = {out:?}");
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
        let mut out = [0u8; five8::BASE58_ENCODED_32_MAX_LEN];
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
            let len = five8::encode_32(&pubkey_bytes, &mut out);
            let pubkey = str::from_utf8(&out[..len as usize]).unwrap();
            let out_str_target_check = maybe_bs58_aware_lowercase(pubkey, args.case_insensitive);
            
            count += 1;
            
            // Did cpu find target?
            if out_str_target_check.starts_with(target) {
                let time_secs = timer.elapsed().as_secs_f64();
                logfather::info!(
                    "cpu {i} found target: {pubkey}; {seed:?} in {:.3}s; {} attempts; {} attempts per second",
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

fn get_validated_target(args: &Args) -> &'static str {
    // Static string of BS58 characters
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Validate target (i.e. does it include 0, O, I, l)
    //
    // maybe TODO: technically we could accept I or o if case-insensitivity but I suspect
    // most users will provide lowercase targets for case-insensitive searches
    for c in args.target.chars() {
        assert!(
            BS58_CHARS.contains(c),
            "your target contains invalid bs58: {}",
            c
        );
    }

    // bs58-aware lowercase converison
    let target = maybe_bs58_aware_lowercase(&args.target, args.case_insensitive);

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

fn parse_pubkey(input: &str) -> Result<[u8; 32], String> {
    let mut out = [0u8; 32];
    five8::decode_32(input, &mut out).map_err(|e| format!("{e:?}"))?;
    Ok(out)
}

fn maybe_update_num_cpus(num_cpus: &mut u32) {
    if *num_cpus == 0 {
        *num_cpus = rayon::current_num_threads() as u32;
    }
}
