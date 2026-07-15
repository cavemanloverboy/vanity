mod fast;

use clap::Parser;
use ed25519_dalek::SigningKey;
use num_bigint::BigUint;
use num_format::{Locale, ToFormattedString};
use num_traits::{One, ToPrimitive, Zero};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use sha2::{Digest, Sha256};
use solana_pubkey::Pubkey;
#[cfg(feature = "deploy")]
use {
    solana_rpc_client::rpc_client::RpcClient,
    solana_sdk::{
        bpf_loader_upgradeable::{self, get_program_data_address, UpgradeableLoaderState},
        instruction::{AccountMeta, Instruction},
        loader_upgradeable_instruction::UpgradeableLoaderInstruction,
        signature::read_keypair_file,
        signer::Signer,
        system_instruction, system_program, sysvar,
        transaction::Transaction,
    },
    std::path::PathBuf,
};

use std::{
    array,
    io::Write,
    str::FromStr,
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Parser)]
pub enum Command {
    Grind(GrindArgs),
    GrindKeypair(GrindKeypairArgs),
    GrindDoppler(DopplerArgs),
    Verify(VerifyArgs),
    #[cfg(feature = "deploy")]
    Deploy(DeployArgs),
}

#[derive(Debug, Parser)]
#[command(group(
    clap::ArgGroup::new("target")
        .required(true)
        .multiple(true)
        .args(["prefix", "suffix"])
))]
pub struct GrindArgs {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: Pubkey,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// Target prefixes for the pubkey (repeat or separate with commas)
    #[clap(long, value_delimiter = ',', value_parser = parse_bs58_pattern)]
    pub prefix: Vec<String>,

    /// Target suffixes for the pubkey (repeat or separate with commas)
    #[clap(long, value_delimiter = ',', value_parser = parse_bs58_pattern)]
    pub suffix: Vec<String>,

    /// Whether user cares about the case of the pubkey
    #[clap(long, default_value_t = false)]
    pub case_insensitive: bool,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,

    /// Number of matching addresses to find before stopping
    #[clap(long, default_value_t = 1)]
    pub count: u32,
}

#[derive(Debug, Parser)]
#[command(group(
    clap::ArgGroup::new("target")
        .required(true)
        .multiple(true)
        .args(["prefix", "suffix"])
))]
pub struct GrindKeypairArgs {
    /// Target prefixes for the pubkey (repeat or separate with commas)
    #[clap(long, value_delimiter = ',', value_parser = parse_bs58_pattern)]
    pub prefix: Vec<String>,

    /// Target suffixes for the pubkey (repeat or separate with commas)
    #[clap(long, value_delimiter = ',', value_parser = parse_bs58_pattern)]
    pub suffix: Vec<String>,

    /// Whether user cares about the case of the pubkey
    #[clap(long, default_value_t = false)]
    pub case_insensitive: bool,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,

    /// Number of matching keypairs to find before stopping
    #[clap(long, default_value_t = 1)]
    pub count: u32,
}

#[derive(Debug, Parser)]
pub struct DopplerArgs {
    /// How many of the four 8-byte pubkey segments must be sign-extendable
    /// 32-bit values (1-4). Higher is exponentially rarer.
    #[clap(long, default_value_t = 1)]
    pub segments: u8,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,

    /// Number of matching keypairs to find before stopping
    #[clap(long, default_value_t = 1)]
    pub count: u32,
}

#[derive(Debug, Parser)]
pub struct VerifyArgs {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: Pubkey,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    /// The seed to verify
    #[clap(long)]
    pub seed: String,
}

#[cfg(feature = "deploy")]
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
}

// ─── globals ────────────────────────────────────────────────────────────────

static FOUND: AtomicU32 = AtomicU32::new(0);
static TOTAL_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
/// Set by the Ctrl-C handler; makes every grind loop wind down. Needed
/// because with no explicit handler, SIGINT relies on the kernel's default
/// terminate action — which does not apply when the process is PID 1 in a
/// container (common on GPU cloud hosts), so Ctrl-C would otherwise be ignored.
static ABORTED: AtomicBool = AtomicBool::new(false);

fn done(target: u32) -> bool {
    FOUND.load(Ordering::SeqCst) >= target || ABORTED.load(Ordering::SeqCst)
}

// ─── bs58 probability (from cavemanloverboy/bs58p) ──────────────────────────

const BS58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const MAX_BS58_ADDRESS_LEN: usize = 44;
#[cfg(feature = "gpu")]
const MAX_GPU_PATTERNS: usize = 32;
#[cfg(feature = "gpu")]
const MATCH_PLAN_MASKS: usize = MAX_BS58_ADDRESS_LEN * 58;
#[cfg(feature = "gpu")]
const MATCH_PLAN_WORDS: usize = 1 + MATCH_PLAN_MASKS + MAX_BS58_ADDRESS_LEN;

#[derive(Debug)]
struct SearchTargets {
    prefixes: Vec<String>,
    suffixes: Vec<String>,
    case_insensitive: bool,
}

impl SearchTargets {
    fn new(
        prefixes: Vec<String>,
        suffixes: Vec<String>,
        case_insensitive: bool,
    ) -> Self {
        Self {
            prefixes: normalize_patterns(prefixes, true, case_insensitive),
            suffixes: normalize_patterns(suffixes, false, case_insensitive),
            case_insensitive,
        }
    }

    fn probability(&self) -> f64 {
        let mut probability = 0.0;
        if self.prefixes.is_empty() {
            for suffix in &self.suffixes {
                probability +=
                    bs58_probability("", suffix, self.case_insensitive);
            }
        } else if self.suffixes.is_empty() {
            for prefix in &self.prefixes {
                probability +=
                    bs58_probability(prefix, "", self.case_insensitive);
            }
        } else {
            for prefix in &self.prefixes {
                for suffix in &self.suffixes {
                    probability += bs58_probability(
                        prefix,
                        suffix,
                        self.case_insensitive,
                    );
                }
            }
        }
        probability.min(1.0)
    }

    fn expected_attempts(&self) -> f64 {
        let probability = self.probability();
        if probability <= 0.0 {
            f64::INFINITY
        } else {
            1.0 / probability
        }
    }

    fn matches(&self, pubkey: &str) -> bool {
        let prefix_matches = self.prefixes.is_empty()
            || self.prefixes.iter().any(|prefix| {
                matches_pattern(pubkey, prefix, true, self.case_insensitive)
            });
        prefix_matches
            && (self.suffixes.is_empty()
                || self.suffixes.iter().any(|suffix| {
                    matches_pattern(
                        pubkey,
                        suffix,
                        false,
                        self.case_insensitive,
                    )
                }))
    }

    #[cfg(feature = "gpu")]
    fn packed_prefixes(&self) -> Vec<u32> {
        build_match_plan(&self.prefixes, true, self.case_insensitive)
    }

    #[cfg(feature = "gpu")]
    fn packed_suffixes(&self) -> Vec<u32> {
        build_match_plan(&self.suffixes, false, self.case_insensitive)
    }

    #[cfg(feature = "gpu")]
    fn validate_gpu_pattern_count(&self) -> Result<(), String> {
        if self.prefixes.len() > MAX_GPU_PATTERNS
            || self.suffixes.len() > MAX_GPU_PATTERNS
        {
            return Err(format!(
                "GPU searches support at most {MAX_GPU_PATTERNS} non-redundant prefixes and suffixes"
            ));
        }
        Ok(())
    }
}

fn normalize_patterns(
    patterns: Vec<String>,
    prefix: bool,
    case_insensitive: bool,
) -> Vec<String> {
    let mut patterns: Vec<String> = patterns
        .into_iter()
        .map(|pattern| {
            maybe_bs58_aware_lowercase(&pattern, case_insensitive)
        })
        .collect();
    patterns.sort_by_key(String::len);

    let mut normalized: Vec<String> = Vec::with_capacity(patterns.len());
    for pattern in patterns {
        let redundant = normalized.iter().any(|existing| {
            if prefix {
                pattern.starts_with(existing)
            } else {
                pattern.ends_with(existing)
            }
        });
        if !redundant {
            normalized.push(pattern);
        }
    }
    normalized
}

#[cfg(feature = "gpu")]
fn build_match_plan(
    patterns: &[String],
    prefix: bool,
    case_insensitive: bool,
) -> Vec<u32> {
    debug_assert!(
        patterns.len() <= MAX_GPU_PATTERNS,
        "GPU searches support at most {MAX_GPU_PATTERNS} prefixes and suffixes"
    );

    let alphabet = if case_insensitive {
        b"123456789abcdefghjkLmnpqrstuvwxyzabcdefghijkmnopqrstuvwxyz"
    } else {
        BS58_ALPHABET.as_bytes()
    };
    let mut plan = vec![0u32; MATCH_PLAN_WORDS];
    plan[0] = match patterns.len() {
        0 => 0,
        MAX_GPU_PATTERNS => u32::MAX,
        count => (1u32 << count) - 1,
    };
    for (pattern_index, pattern) in patterns.iter().enumerate() {
        let candidate = 1u32 << pattern_index;
        let bytes: Box<dyn Iterator<Item = u8>> = if prefix {
            Box::new(pattern.bytes())
        } else {
            Box::new(pattern.bytes().rev())
        };
        for (position, byte) in bytes.enumerate() {
            let symbol = alphabet
                .iter()
                .position(|&candidate| candidate == byte)
                .unwrap();
            plan[1 + position * 58 + symbol] |= candidate;
            if position + 1 == pattern.len() {
                plan[1 + MATCH_PLAN_MASKS + position] |= candidate;
            }
        }
    }
    plan
}

fn bs58_pure_prefix_suffix_prob(prefix: &str, suffix: &str, n_bytes: usize) -> f64 {
    if prefix.is_empty() && suffix.is_empty() {
        return 1.0;
    }
    let b58 = BigUint::from(58u32);
    let prefix_len = prefix.len();
    let suffix_len = suffix.len();

    let mut p_val = BigUint::zero();
    for (i, c) in prefix.chars().enumerate() {
        let idx = BS58_ALPHABET.find(c).unwrap();
        p_val += BigUint::from(idx) * b58.pow((prefix_len - 1 - i) as u32);
    }
    let mut s_val = BigUint::zero();
    for (i, c) in suffix.chars().enumerate() {
        let idx = BS58_ALPHABET.find(c).unwrap();
        s_val += BigUint::from(idx) * b58.pow((suffix_len - 1 - i) as u32);
    }

    let m_big = BigUint::one() << (8 * n_bytes);
    let ln2 = std::f64::consts::LN_2;
    let ln58 = 58f64.ln();
    let bits = (8 * n_bytes) as f64;
    let l_max = (bits * ln2 / ln58).ceil() as usize;

    let mut total = BigUint::zero();
    let modulus = b58.pow(suffix_len as u32);
    let start_l = std::cmp::max(std::cmp::max(prefix_len, suffix_len), 1);

    for l in start_l..=l_max {
        let pow_lk = b58.pow((l - prefix_len) as u32);
        let low1 = &p_val * &pow_lk;
        let low2 = b58.pow((l - 1) as u32);
        let low_pref = if low1 > low2 { low1 } else { low2 };

        let high1 = (&p_val + BigUint::one()) * &pow_lk;
        let high2 = b58.pow(l as u32);
        let mut high_pref = if high1 < high2 { high1 } else { high2 };
        if high_pref > m_big {
            high_pref = m_big.clone();
        }
        if high_pref <= low_pref {
            continue;
        }
        let r0 = &low_pref % &modulus;
        let delta = (&s_val + &modulus - &r0) % &modulus;
        let first = &low_pref + delta;
        if first >= high_pref {
            continue;
        }
        let cnt = BigUint::one() + (&high_pref - BigUint::one() - &first) / &modulus;
        total += cnt;
    }

    total.to_f64().unwrap() / m_big.to_f64().unwrap()
}

/// How many distinct base58 alphabet symbols match `pattern_c` under the same
/// rules as `bs58_ci_matches` (uppercase `L` in the pattern is literal only).
fn bs58_ci_position_factor(pattern_c: char) -> f64 {
    if !pattern_c.is_ascii_alphabetic() {
        return 1.0;
    }
    let count = BS58_ALPHABET
        .chars()
        .filter(|&a| {
            if pattern_c == 'L' {
                a == 'L'
            } else {
                a.eq_ignore_ascii_case(&pattern_c)
            }
        })
        .count();
    f64::from(count.max(1) as u32)
}

fn bs58_ci_factor(prefix: &str, suffix: &str) -> f64 {
    prefix
        .chars()
        .chain(suffix.chars())
        .map(bs58_ci_position_factor)
        .product()
}

fn bs58_probability(prefix: &str, suffix: &str, case_insensitive: bool) -> f64 {
    let zeros = prefix.chars().take_while(|&c| c == '1').count();
    let pre_nz = &prefix[zeros..];
    let p_zero = if pre_nz.is_empty() {
        (1.0_f64 / 256.0).powi(zeros as i32)
    } else {
        (1.0_f64 / 256.0).powi(zeros as i32) * (255.0 / 256.0)
    };
    let rem = 32_usize.saturating_sub(zeros);
    let pure = bs58_pure_prefix_suffix_prob(pre_nz, suffix, rem);
    let prob = p_zero * pure;

    if case_insensitive {
        prob * bs58_ci_factor(prefix, suffix)
    } else {
        prob
    }
}

// ─── formatting ─────────────────────────────────────────────────────────────

fn format_duration(secs: f64) -> String {
    if secs < 0.0 {
        return "any moment".into();
    }
    if secs < 60.0 {
        return format!("{:.0}s", secs);
    }
    let s = secs as u64;
    if s < 3600 {
        return format!("{}m {}s", s / 60, s % 60);
    }
    if s < 86400 {
        return format!("{}h {}m", s / 3600, (s % 3600) / 60);
    }
    format!("{}d {}h", s / 86400, (s % 86400) / 3600)
}

fn print_status(total: u64, rate: f64, elapsed: f64, expected: f64) {
    let e_time = if rate > 0.0 && expected.is_finite() {
        format!(" | E[grind_time] = {}", format_duration(expected / rate))
    } else {
        String::new()
    };
    eprint!(
        "\r\x1b[K{} attempts | {} attempts/sec | elapsed: {}{}",
        total.to_formatted_string(&Locale::en),
        (rate as u64).to_formatted_string(&Locale::en),
        format_duration(elapsed),
        e_time,
    );
    let _ = std::io::stderr().flush();
}

fn spawn_hashrate_reporter(
    shutdown: Arc<AtomicBool>,
    expected: f64,
    start: Instant,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_secs(1));
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        let elapsed = start.elapsed().as_secs_f64();
        let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
        let rate = total as f64 / elapsed.max(1e-9);
        print_status(total, rate, elapsed, expected);
    })
}

// ─── main ───────────────────────────────────────────────────────────────────

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    // Explicit Ctrl-C handler. First press winds the grind down cleanly (so
    // the final stats print and GPU contexts are released); a second press
    // forces an immediate exit. Without this, SIGINT is ignored when the
    // process is PID 1 in a container (e.g. GPU cloud hosts).
    let _ = ctrlc::set_handler(|| {
        if ABORTED.swap(true, Ordering::SeqCst) {
            std::process::exit(130);
        }
        fast::request_abort();
        eprintln!("\naborting… (press Ctrl-C again to force-quit)");
    });

    let command = Command::parse();
    match command {
        Command::Grind(args) => grind(args),
        Command::GrindKeypair(args) => grind_keypair(args),
        Command::GrindDoppler(args) => grind_doppler(args),
        Command::Verify(args) => verify(args),
        #[cfg(feature = "deploy")]
        Command::Deploy(args) => deploy(args),
    }
}

fn verify(args: VerifyArgs) {
    let VerifyArgs { base, owner, seed } = args;
    let result = Pubkey::create_with_seed(&base, &seed, &owner).unwrap();
    println!("Results:");
    println!("  base  {base}");
    println!("  owner {owner}");
    println!("  seed  {seed}\n");
    println!("  resulting pubkey: {result}")
}

#[cfg(feature = "deploy")]
fn deploy(args: DeployArgs) {
    let base_keypair = read_keypair_file(&args.base).expect("failed to read base keypair");
    let payer_keypair = args
        .payer
        .as_ref()
        .map(|payer| read_keypair_file(payer).expect("failed to read payer keypair"))
        .unwrap_or(base_keypair.insecure_clone());
    let authority = args.authority.unwrap_or_else(|| payer_keypair.pubkey());

    let target = Pubkey::create_with_seed(&base_keypair.pubkey(), &args.seed, &args.owner).unwrap();
    let rpc_client = RpcClient::new(args.rpc);
    let buffer_len = rpc_client.get_account_data(&args.buffer).unwrap().len();
    let rent = rpc_client
        .get_minimum_balance_for_rent_exemption(UpgradeableLoaderState::size_of_program())
        .expect("failed to fetch rent");

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

#[cfg(feature = "deploy")]
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

// ─── grind ──────────────────────────────────────────────────────────────────

fn grind(mut args: GrindArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);
    let targets = Arc::new(SearchTargets::new(
        std::mem::take(&mut args.prefix),
        std::mem::take(&mut args.suffix),
        args.case_insensitive,
    ));
    #[cfg(feature = "gpu")]
    if args.num_gpus > 0 {
        targets.validate_gpu_pattern_count().unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(2);
        });
    }

    let expected = targets.expected_attempts();
    let prob = targets.probability();
    #[cfg(feature = "gpu")]
    eprintln!("using {} cpus, {} gpus", args.num_cpus, args.num_gpus);
    #[cfg(not(feature = "gpu"))]
    eprintln!("using {} cpus", args.num_cpus);
    let target_label = format_target_label(&targets);
    eprintln!(
        "target: {} | probability: {:.6e} | expected: {} attempts",
        target_label,
        prob,
        (expected as u64).to_formatted_string(&Locale::en)
    );

    let target_count = args.count;
    let shutdown = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let base = args.base;
        let owner = args.owner;
        let ci = args.case_insensitive;
        let targets = Arc::clone(&targets);
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    let prefixes = targets.packed_prefixes();
                    let suffixes = targets.packed_suffixes();
                    let mut contexts = Vec::with_capacity(num_gpus as usize);
                    for id in 0..num_gpus {
                        let ctx = unsafe {
                            gpu_grind_init(
                                id as i32,
                                base.as_ref().as_ptr(),
                                owner.as_ref().as_ptr(),
                                prefixes.as_ptr().cast(),
                                targets.prefixes.len() as u64,
                                suffixes.as_ptr().cast(),
                                targets.suffixes.len() as u64,
                                ci,
                            )
                        };
                        contexts.push(ctx);
                    }

                    let mut iterations = vec![0u64; num_gpus as usize];
                    let mut launch_times = vec![Instant::now(); num_gpus as usize];
                    let mut in_flight = vec![false; num_gpus as usize];

                    for (i, &ctx) in contexts.iter().enumerate() {
                        let seed = new_gpu_seed(i as u32, 0);
                        launch_times[i] = Instant::now();
                        unsafe {
                            gpu_grind_launch(ctx, seed.as_ptr());
                        }
                        in_flight[i] = true;
                    }

                    loop {
                        if done(target_count) {
                            break;
                        }

                        let mut any_ready = false;
                        for (i, &ctx) in contexts.iter().enumerate() {
                            if !in_flight[i] {
                                continue;
                            }
                            if unsafe { gpu_grind_query(ctx) } == 0 {
                                continue;
                            }
                            any_ready = true;

                            let time_sec = launch_times[i].elapsed().as_secs_f64();
                            let mut out = [0u8; 24];
                            unsafe {
                                gpu_grind_read(ctx, out.as_mut_ptr());
                            }

                            let count = u64::from_le_bytes(array::from_fn(|j| out[16 + j]));
                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);

                            let reconstructed: [u8; 32] = Sha256::new()
                                .chain_update(base)
                                .chain_update(&out[..16])
                                .chain_update(owner)
                                .finalize()
                                .into();
                            let out_str = fd_bs58::encode_32(reconstructed);

                            if targets.matches(&out_str) {
                                eprintln!(
                                    "\r\x1b[Kgpu {} match: {} in {:.3}s",
                                    i, &out_str, time_sec
                                );
                                eprintln!(
                                    "out seed = {out:?} -> {}",
                                    core::str::from_utf8(&out[..16]).unwrap()
                                );
                                FOUND.fetch_add(1, Ordering::SeqCst);
                            }

                            in_flight[i] = false;
                            if !done(target_count) {
                                iterations[i] += 1;
                                let seed = new_gpu_seed(i as u32, iterations[i]);
                                launch_times[i] = Instant::now();
                                unsafe {
                                    gpu_grind_launch(ctx, seed.as_ptr());
                                }
                                in_flight[i] = true;
                            }
                        }

                        if !any_ready {
                            thread::sleep(Duration::from_millis(10));
                        }
                    }

                    for (i, &ctx) in contexts.iter().enumerate() {
                        if in_flight[i] {
                            while unsafe { gpu_grind_query(ctx) } == 0 {
                                thread::sleep(Duration::from_millis(10));
                            }
                            let mut out = [0u8; 24];
                            unsafe {
                                gpu_grind_read(ctx, out.as_mut_ptr());
                            }
                            let count = u64::from_le_bytes(array::from_fn(|j| out[16 + j]));
                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);
                        }
                    }
                    for ctx in contexts {
                        unsafe {
                            gpu_grind_destroy(ctx);
                        }
                    }
                })
                .unwrap(),
        )
    } else {
        None
    };

    let grind_start = Instant::now();
    let reporter = spawn_hashrate_reporter(Arc::clone(&shutdown), expected, grind_start);

    (0..args.num_cpus).into_par_iter().for_each(|i| {
        let timer = Instant::now();
        let mut local_batch = 0_u64;

        let base_sha = Sha256::new().chain_update(args.base);
        loop {
            if done(target_count) {
                if local_batch > 0 {
                    TOTAL_ATTEMPTS.fetch_add(local_batch, Ordering::Relaxed);
                }
                return;
            }

            let seed: [u8; 16] = rand::random();
            let seed: [u8; 16] = array::from_fn(|i| {
                const ALNUM: &[u8] =
                    b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
                ALNUM[seed[i] as usize % ALNUM.len()]
            });

            let pubkey_bytes: [u8; 32] = base_sha
                .clone()
                .chain_update(seed)
                .chain_update(args.owner)
                .finalize()
                .into();
            let pubkey = fd_bs58::encode_32(pubkey_bytes);

            local_batch += 1;
            if local_batch >= 4096 {
                TOTAL_ATTEMPTS.fetch_add(4096, Ordering::Relaxed);
                local_batch -= 4096;
            }

            if targets.matches(&pubkey) {
                if local_batch > 0 {
                    TOTAL_ATTEMPTS.fetch_add(local_batch, Ordering::Relaxed);
                    local_batch = 0;
                }
                let time_secs = timer.elapsed().as_secs_f64();
                let elapsed_global = grind_start.elapsed().as_secs_f64().max(1e-9);
                let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
                let global_rate = total as f64 / elapsed_global;
                eprintln!(
                    "\r\x1b[Kcpu {i} match: {pubkey}; {seed:?} -> {} in {:.3}s; {} attempts/sec",
                    core::str::from_utf8(&seed).unwrap(),
                    time_secs,
                    (global_rate as u64).to_formatted_string(&Locale::en)
                );
                FOUND.fetch_add(1, Ordering::SeqCst);
                if done(target_count) {
                    break;
                }
            }
        }
    });

    #[cfg(feature = "gpu")]
    if let Some(t) = gpu_thread {
        t.join().unwrap();
    }

    shutdown.store(true, Ordering::SeqCst);
    reporter.join().unwrap();

    let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
    let elapsed = grind_start.elapsed().as_secs_f64().max(1e-9);
    let rate = total as f64 / elapsed;
    eprintln!(
        "\r\x1b[Kdone: {} attempts in {} at {} attempts/sec",
        total.to_formatted_string(&Locale::en),
        format_duration(elapsed),
        (rate as u64).to_formatted_string(&Locale::en)
    );
}

// ─── grind-keypair ──────────────────────────────────────────────────────────

fn grind_keypair(mut args: GrindKeypairArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);
    let targets = Arc::new(SearchTargets::new(
        std::mem::take(&mut args.prefix),
        std::mem::take(&mut args.suffix),
        args.case_insensitive,
    ));
    #[cfg(feature = "gpu")]
    if args.num_gpus > 0 {
        targets.validate_gpu_pattern_count().unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(2);
        });
    }

    let expected = targets.expected_attempts();
    let prob = targets.probability();
    #[cfg(feature = "gpu")]
    eprintln!(
        "using {} cpus, {} gpus (cpu backend: {})",
        args.num_cpus,
        args.num_gpus,
        fast::backend_name()
    );
    #[cfg(not(feature = "gpu"))]
    eprintln!(
        "using {} cpus (cpu backend: {})",
        args.num_cpus,
        fast::backend_name()
    );
    let target_label = format_target_label(&targets);
    eprintln!(
        "target: {} | probability: {:.6e} | expected: {} attempts",
        target_label,
        prob,
        (expected as u64).to_formatted_string(&Locale::en)
    );

    let target_count = args.count;
    fast::reset_grind();
    let shutdown = Arc::new(AtomicBool::new(false));
    let grind_start = Instant::now();

    // Reporter reads the shared fast-path counters (CPU + GPU both update them).
    let reporter = {
        let shutdown = Arc::clone(&shutdown);
        thread::spawn(move || loop {
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            thread::sleep(Duration::from_secs(1));
            if shutdown.load(Ordering::SeqCst) {
                break;
            }
            let elapsed = grind_start.elapsed().as_secs_f64();
            let total = fast::total_attempts();
            let rate = total as f64 / elapsed.max(1e-9);
            print_status(total, rate, elapsed, expected);
        })
    };

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let ci = args.case_insensitive;
        let targets = Arc::clone(&targets);
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    let prefixes = targets.packed_prefixes();
                    let suffixes = targets.packed_suffixes();
                    let mut contexts = Vec::with_capacity(num_gpus as usize);
                    for id in 0..num_gpus {
                        let ctx = unsafe {
                            gpu_keypair_init(
                                id as i32,
                                prefixes.as_ptr().cast(),
                                targets.prefixes.len() as u64,
                                suffixes.as_ptr().cast(),
                                targets.suffixes.len() as u64,
                                ci,
                            )
                        };
                        contexts.push(ctx);
                    }

                    let mut iterations = vec![0u64; num_gpus as usize];
                    let mut launch_times = vec![Instant::now(); num_gpus as usize];
                    let mut in_flight = vec![false; num_gpus as usize];

                    for (i, &ctx) in contexts.iter().enumerate() {
                        let seed = new_gpu_seed(i as u32, 0);
                        launch_times[i] = Instant::now();
                        unsafe {
                            gpu_keypair_launch(ctx, seed.as_ptr());
                        }
                        in_flight[i] = true;
                    }

                    loop {
                        if fast::is_done(target_count) || ABORTED.load(Ordering::Relaxed) {
                            break;
                        }

                        let mut any_ready = false;
                        for (i, &ctx) in contexts.iter().enumerate() {
                            if !in_flight[i] {
                                continue;
                            }
                            if unsafe { gpu_keypair_query(ctx) } == 0 {
                                continue;
                            }
                            any_ready = true;

                            let time_sec = launch_times[i].elapsed().as_secs_f64();
                            let mut out = [0u8; 40];
                            unsafe {
                                gpu_keypair_read(ctx, out.as_mut_ptr());
                            }

                            let found_seed: [u8; 32] = out[..32].try_into().unwrap();
                            let signing_key = SigningKey::from_bytes(&found_seed);
                            let pubkey_bytes = signing_key.verifying_key().to_bytes();
                            let pubkey_str = fd_bs58::encode_32(pubkey_bytes);
                            let count = u64::from_le_bytes(array::from_fn(|j| out[32 + j]));

                            fast::add_attempts(count);

                            if targets.matches(&pubkey_str) {
                                let prev = fast::note_found();
                                if prev < target_count {
                                    eprintln!(
                                        "\r\x1b[Kgpu {} match: {} in {:.3}s",
                                        i, &pubkey_str, time_sec
                                    );
                                    print_keypair_result(
                                        &found_seed,
                                        &pubkey_bytes,
                                        &pubkey_str,
                                    );
                                }
                            }

                            in_flight[i] = false;
                            if !fast::is_done(target_count) && !ABORTED.load(Ordering::Relaxed) {
                                iterations[i] += 1;
                                let seed = new_gpu_seed(i as u32, iterations[i]);
                                launch_times[i] = Instant::now();
                                unsafe {
                                    gpu_keypair_launch(ctx, seed.as_ptr());
                                }
                                in_flight[i] = true;
                            }
                        }

                        if !any_ready {
                            thread::sleep(Duration::from_millis(10));
                        }
                    }

                    for (i, &ctx) in contexts.iter().enumerate() {
                        if in_flight[i] {
                            while unsafe { gpu_keypair_query(ctx) } == 0 {
                                thread::sleep(Duration::from_millis(10));
                            }
                            let mut out = [0u8; 40];
                            unsafe {
                                gpu_keypair_read(ctx, out.as_mut_ptr());
                            }
                            let count = u64::from_le_bytes(array::from_fn(|j| out[32 + j]));
                            fast::add_attempts(count);
                        }
                    }
                    for ctx in contexts {
                        unsafe {
                            gpu_keypair_destroy(ctx);
                        }
                    }
                })
                .unwrap(),
        )
    } else {
        None
    };

    fast::run_cpu_workers(
        &targets.prefixes,
        &targets.suffixes,
        args.case_insensitive,
        args.num_cpus,
        target_count,
    );

    // CPU workers finished (found enough or aborted); stop GPU too.
    fast::request_abort();

    #[cfg(feature = "gpu")]
    if let Some(t) = gpu_thread {
        t.join().unwrap();
    }

    shutdown.store(true, Ordering::SeqCst);
    reporter.join().unwrap();

    let total = fast::total_attempts();
    let elapsed = grind_start.elapsed().as_secs_f64().max(1e-9);
    let rate = total as f64 / elapsed;
    eprintln!(
        "\r\x1b[Kdone: {} attempts in {} at {} attempts/sec",
        total.to_formatted_string(&Locale::en),
        format_duration(elapsed),
        (rate as u64).to_formatted_string(&Locale::en)
    );
}

// ─── doppler ──────────────────────────────────────────────────────────────

fn grind_doppler(mut args: DopplerArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);
    assert!(
        (1..=4).contains(&args.segments),
        "--segments must be between 1 and 4"
    );

    let prob = doppler_probability(args.segments);
    let expected = if prob > 0.0 {
        1.0 / prob
    } else {
        f64::INFINITY
    };

    #[cfg(feature = "gpu")]
    eprintln!("using {} cpus, {} gpus", args.num_cpus, args.num_gpus);
    #[cfg(not(feature = "gpu"))]
    eprintln!("using {} cpus", args.num_cpus);
    eprintln!(
        "doppler: >= {} sign-extendable 32-bit segment(s) | probability: {:.6e} | expected: {} attempts",
        args.segments,
        prob,
        (expected as u64).to_formatted_string(&Locale::en)
    );

    let target_count = args.count;
    let shutdown = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let segments = args.segments as u32;
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    let mut contexts = Vec::with_capacity(num_gpus as usize);
                    for id in 0..num_gpus {
                        let ctx = unsafe { gpu_doppler_init(id as i32, segments) };
                        contexts.push(ctx);
                    }

                    let mut iterations = vec![0u64; num_gpus as usize];
                    let mut launch_times = vec![Instant::now(); num_gpus as usize];
                    let mut in_flight = vec![false; num_gpus as usize];

                    for (i, &ctx) in contexts.iter().enumerate() {
                        let seed = new_gpu_seed(i as u32, 0);
                        launch_times[i] = Instant::now();
                        unsafe {
                            gpu_doppler_launch(ctx, seed.as_ptr());
                        }
                        in_flight[i] = true;
                    }

                    loop {
                        if done(target_count) {
                            break;
                        }

                        let mut any_ready = false;
                        for (i, &ctx) in contexts.iter().enumerate() {
                            if !in_flight[i] {
                                continue;
                            }
                            if unsafe { gpu_doppler_query(ctx) } == 0 {
                                continue;
                            }
                            any_ready = true;

                            let time_sec = launch_times[i].elapsed().as_secs_f64();
                            let mut out = [0u8; 40];
                            unsafe {
                                gpu_doppler_read(ctx, out.as_mut_ptr());
                            }

                            let found_seed: [u8; 32] = out[..32].try_into().unwrap();
                            let signing_key = SigningKey::from_bytes(&found_seed);
                            let pubkey_bytes = signing_key.verifying_key().to_bytes();
                            let count = u64::from_le_bytes(array::from_fn(|j| out[32 + j]));

                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);

                            if doppler_count_segments(&pubkey_bytes) >= segments {
                                let pubkey_str = fd_bs58::encode_32(pubkey_bytes);
                                eprintln!(
                                    "\r\x1b[Kgpu {} match: {} in {:.3}s",
                                    i, &pubkey_str, time_sec
                                );
                                print_doppler_result(&found_seed, &pubkey_bytes, &pubkey_str);
                                FOUND.fetch_add(1, Ordering::SeqCst);
                            }

                            in_flight[i] = false;
                            if !done(target_count) {
                                iterations[i] += 1;
                                let seed = new_gpu_seed(i as u32, iterations[i]);
                                launch_times[i] = Instant::now();
                                unsafe {
                                    gpu_doppler_launch(ctx, seed.as_ptr());
                                }
                                in_flight[i] = true;
                            }
                        }

                        if !any_ready {
                            thread::sleep(Duration::from_millis(10));
                        }
                    }

                    for (i, &ctx) in contexts.iter().enumerate() {
                        if in_flight[i] {
                            while unsafe { gpu_doppler_query(ctx) } == 0 {
                                thread::sleep(Duration::from_millis(10));
                            }
                            let mut out = [0u8; 40];
                            unsafe {
                                gpu_doppler_read(ctx, out.as_mut_ptr());
                            }
                            let count = u64::from_le_bytes(array::from_fn(|j| out[32 + j]));
                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);
                        }
                    }
                    for ctx in contexts {
                        unsafe {
                            gpu_doppler_destroy(ctx);
                        }
                    }
                })
                .unwrap(),
        )
    } else {
        None
    };

    let grind_start = Instant::now();
    let reporter = spawn_hashrate_reporter(Arc::clone(&shutdown), expected, grind_start);

    let segments = args.segments as u32;
    (0..args.num_cpus).into_par_iter().for_each(|i| {
        let timer = Instant::now();
        let mut local_batch = 0_u64;

        loop {
            if done(target_count) {
                if local_batch > 0 {
                    TOTAL_ATTEMPTS.fetch_add(local_batch, Ordering::Relaxed);
                }
                return;
            }

            let seed: [u8; 32] = rand::random();
            let signing_key = SigningKey::from_bytes(&seed);
            let pubkey_bytes = signing_key.verifying_key().to_bytes();

            local_batch += 1;
            if local_batch >= 4096 {
                TOTAL_ATTEMPTS.fetch_add(4096, Ordering::Relaxed);
                local_batch -= 4096;
            }

            if doppler_count_segments(&pubkey_bytes) >= segments {
                if local_batch > 0 {
                    TOTAL_ATTEMPTS.fetch_add(local_batch, Ordering::Relaxed);
                    local_batch = 0;
                }
                let pubkey_str = fd_bs58::encode_32(pubkey_bytes);
                let time_secs = timer.elapsed().as_secs_f64();
                let elapsed_global = grind_start.elapsed().as_secs_f64().max(1e-9);
                let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
                let global_rate = total as f64 / elapsed_global;
                eprintln!(
                    "\r\x1b[Kcpu {i} match: {pubkey_str} in {:.3}s; {} attempts/sec",
                    time_secs,
                    (global_rate as u64).to_formatted_string(&Locale::en)
                );
                print_doppler_result(&seed, &pubkey_bytes, &pubkey_str);
                FOUND.fetch_add(1, Ordering::SeqCst);
                if done(target_count) {
                    break;
                }
            }
        }
    });

    #[cfg(feature = "gpu")]
    if let Some(t) = gpu_thread {
        t.join().unwrap();
    }

    shutdown.store(true, Ordering::SeqCst);
    reporter.join().unwrap();

    let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
    let elapsed = grind_start.elapsed().as_secs_f64().max(1e-9);
    let rate = total as f64 / elapsed;
    eprintln!(
        "\r\x1b[Kdone: {} attempts in {} at {} attempts/sec",
        total.to_formatted_string(&Locale::en),
        format_duration(elapsed),
        (rate as u64).to_formatted_string(&Locale::en)
    );
}

// ─── helpers ────────────────────────────────────────────────────────────────

/// Count how many of the four 8-byte pubkey segments are sign-extendable
/// 32-bit values: low 4 bytes are an i32, high 4 bytes are its sign extension.
fn doppler_count_segments(pubkey: &[u8; 32]) -> u32 {
    let mut matched = 0;
    for s in 0..4 {
        let o = s * 8;
        let fill = if pubkey[o + 3] & 0x80 != 0 {
            0xFF
        } else {
            0x00
        };
        if pubkey[o + 4] == fill
            && pubkey[o + 5] == fill
            && pubkey[o + 6] == fill
            && pubkey[o + 7] == fill
        {
            matched += 1;
        }
    }
    matched
}

/// Probability that a uniformly random pubkey has at least `required` of its
/// four segments sign-extendable. Each segment matches with p = 2^-32 (2^32
/// of the 2^64 byte patterns), so this is a binomial tail over 4 trials.
fn doppler_probability(required: u8) -> f64 {
    let p = 2f64.powi(-32);
    let q = 1.0 - p;
    const BINOM4: [f64; 5] = [1.0, 4.0, 6.0, 4.0, 1.0]; // C(4, k)
    let mut total = 0.0;
    for k in (required as u32)..=4 {
        total += BINOM4[k as usize] * p.powi(k as i32) * q.powi((4 - k) as i32);
    }
    total
}

/// Print the matched keypair plus a per-segment breakdown, including the
/// assembly `.equ` constants the doppler-keygen reference emits.
fn print_doppler_result(seed: &[u8; 32], pubkey: &[u8; 32], pubkey_str: &str) {
    print_keypair_result(seed, pubkey, pubkey_str);
    eprintln!(
        "doppler: {}/4 sign-extendable segment(s)",
        doppler_count_segments(pubkey)
    );
    for s in 0..4usize {
        let o = s * 8;
        let fill = if pubkey[o + 3] & 0x80 != 0 {
            0xFF
        } else {
            0x00
        };
        let sign_extendable = pubkey[o + 4] == fill
            && pubkey[o + 5] == fill
            && pubkey[o + 6] == fill
            && pubkey[o + 7] == fill;
        if sign_extendable {
            let imm = i32::from_le_bytes([pubkey[o], pubkey[o + 1], pubkey[o + 2], pubkey[o + 3]]);
            eprintln!(
                "  seg {s} (bytes {}-{}): imm32 {} (0x{:08x})  =>  .equ EXPECTED_KEY_{s}, 0x{:08x}",
                o,
                o + 7,
                imm,
                imm as u32,
                imm as u32
            );
        } else {
            let full = u64::from_le_bytes(array::from_fn(|j| pubkey[o + j]));
            eprintln!(
                "  seg {s} (bytes {}-{}): 0x{:016x} (not sign-extendable)  =>  .equ EXPECTED_KEY_{s}, 0x{:016x}",
                o,
                o + 7,
                full,
                full
            );
        }
    }
}

fn format_target_label(targets: &SearchTargets) -> String {
    fn format_group(label: &str, patterns: &[String]) -> Option<String> {
        match patterns {
            [] => None,
            [pattern] => Some(format!("{label}={pattern}")),
            _ => Some(format!("{label}=[{}]", patterns.join(" | "))),
        }
    }

    [
        format_group("prefix", &targets.prefixes),
        format_group("suffix", &targets.suffixes),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(" AND ")
}

fn print_keypair_result(seed: &[u8; 32], pubkey: &[u8; 32], pubkey_str: &str) {
    let seed_hex: String = seed.iter().map(|b| format!("{b:02x}")).collect();
    eprintln!("pubkey:   {pubkey_str}");
    eprintln!("seed hex: {seed_hex}");
    let keypair_json: Vec<u8> = seed.iter().chain(pubkey.iter()).copied().collect();
    eprintln!("keypair json (solana-compatible): {:?}", keypair_json);
}

fn parse_bs58_pattern(pattern: &str) -> Result<String, String> {
    if pattern.is_empty() {
        return Err("pattern cannot be empty".to_string());
    }
    if pattern.len() > MAX_BS58_ADDRESS_LEN {
        return Err(format!(
            "pattern cannot exceed {MAX_BS58_ADDRESS_LEN} characters"
        ));
    }
    if let Some(invalid) = pattern
        .chars()
        .find(|character| !BS58_ALPHABET.contains(*character))
    {
        return Err(format!("pattern contains invalid base58 character: {invalid}"));
    }
    Ok(pattern.to_string())
}

fn maybe_bs58_aware_lowercase(target: &str, case_insensitive: bool) -> String {
    if case_insensitive {
        target
            .chars()
            .map(|c| if c == 'L' { c } else { c.to_ascii_lowercase() })
            .collect::<String>()
    } else {
        target.to_string()
    }
}

fn matches_pattern(
    pubkey: &str,
    pattern: &str,
    prefix: bool,
    case_insensitive: bool,
) -> bool {
    if case_insensitive {
        bs58_ci_matches(pubkey, pattern, prefix)
    } else if prefix {
        pubkey.starts_with(pattern)
    } else {
        pubkey.ends_with(pattern)
    }
}

fn bs58_ci_matches(haystack: &str, pattern: &str, prefix: bool) -> bool {
    let h = if prefix {
        &haystack[..pattern.len().min(haystack.len())]
    } else {
        let start = haystack.len().saturating_sub(pattern.len());
        &haystack[start..]
    };
    if h.len() != pattern.len() {
        return false;
    }
    h.bytes().zip(pattern.bytes()).all(|(a, b)| {
        if b == b'L' {
            a == b'L'
        } else {
            a.to_ascii_lowercase() == b
        }
    })
}

#[cfg(feature = "gpu")]
extern "C" {
    pub fn gpu_grind_init(
        id: i32,
        base: *const u8,
        owner: *const u8,
        prefixes: *const u8,
        prefix_count: u64,
        suffixes: *const u8,
        suffix_count: u64,
        case_insensitive: bool,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_grind_launch(ctx: *mut std::ffi::c_void, seed: *const u8);
    pub fn gpu_grind_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_grind_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_grind_destroy(ctx: *mut std::ffi::c_void);

    pub fn gpu_keypair_init(
        id: i32,
        prefixes: *const u8,
        prefix_count: u64,
        suffixes: *const u8,
        suffix_count: u64,
        case_insensitive: bool,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_keypair_launch(ctx: *mut std::ffi::c_void, seed: *const u8);
    pub fn gpu_keypair_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_keypair_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_keypair_destroy(ctx: *mut std::ffi::c_void);

    pub fn gpu_doppler_init(id: i32, required_segments: u32) -> *mut std::ffi::c_void;
    pub fn gpu_doppler_launch(ctx: *mut std::ffi::c_void, seed: *const u8);
    pub fn gpu_doppler_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_doppler_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_doppler_destroy(ctx: *mut std::ffi::c_void);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bs58_ci_factor_mithril_counts_i_once() {
        // Base58 has lowercase i but not uppercase I; old code assumed 2^6=64.
        assert_eq!(bs58_ci_factor("mithriL", ""), 16.0);
    }

    #[test]
    fn bs58_ci_factor_literal_l_is_not_fuzzy() {
        assert_eq!(bs58_ci_position_factor('L'), 1.0);
        // Base58 has uppercase L but not lowercase l; pattern l still only maps to L.
        assert_eq!(bs58_ci_position_factor('l'), 1.0);
    }

    #[test]
    fn bs58_ci_factor_skips_non_letters() {
        assert_eq!(bs58_ci_factor("1A", ""), 2.0);
    }

    #[test]
    fn cli_accepts_repeated_and_comma_delimited_targets() {
        let command = Command::try_parse_from([
            "vanity",
            "grind-keypair",
            "--prefix",
            "sun",
            "--prefix",
            "moon,mint",
            "--suffix",
            "key",
        ])
        .unwrap();

        let Command::GrindKeypair(args) = command else {
            panic!("expected grind-keypair command");
        };
        assert_eq!(args.prefix, ["sun", "moon", "mint"]);
        assert_eq!(args.suffix, ["key"]);
    }

    #[test]
    fn cli_requires_a_target() {
        assert!(Command::try_parse_from(["vanity", "grind-keypair"])
            .is_err());
    }

    #[test]
    fn targets_match_any_prefix_and_any_suffix() {
        let targets = SearchTargets::new(
            vec!["sun".to_string(), "moon".to_string()],
            vec!["key".to_string(), "mint".to_string()],
            false,
        );
        assert!(targets.matches("sun7testkey"));
        assert!(targets.matches("moon7testmint"));
        assert!(!targets.matches("sun7testend"));
        assert!(!targets.matches("star7testkey"));
    }

    #[test]
    fn normalization_removes_duplicate_and_redundant_targets() {
        let targets = SearchTargets::new(
            vec!["Sun".to_string(), "sun".to_string(), "sunny".to_string()],
            vec!["mint".to_string(), "int".to_string()],
            true,
        );
        assert_eq!(targets.prefixes, ["sun"]);
        assert_eq!(targets.suffixes, ["int"]);
    }

    #[test]
    fn multi_target_probability_sums_disjoint_alternatives() {
        let targets = SearchTargets::new(
            vec!["sun".to_string(), "moon".to_string()],
            Vec::new(),
            false,
        );
        let expected = bs58_probability("sun", "", false)
            + bs58_probability("moon", "", false);
        assert!((targets.probability() - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn parser_rejects_invalid_or_overlong_patterns() {
        assert!(parse_bs58_pattern("zero0").is_err());
        assert!(parse_bs58_pattern(&"a".repeat(45)).is_err());
    }

    #[cfg(feature = "gpu")]
    fn match_plan_matches(plan: &[u32], value: &str, reverse: bool) -> bool {
        let mut candidates = plan[0];
        let bytes: Box<dyn Iterator<Item = u8>> = if reverse {
            Box::new(value.bytes().rev())
        } else {
            Box::new(value.bytes())
        };
        for (position, byte) in bytes.enumerate() {
            if position >= MAX_BS58_ADDRESS_LEN {
                return false;
            }
            let symbol = BS58_ALPHABET
                .as_bytes()
                .iter()
                .position(|&candidate| candidate == byte)
                .unwrap();
            candidates &= plan[1 + position * 58 + symbol];
            if candidates & plan[1 + MATCH_PLAN_MASKS + position] != 0 {
                return true;
            }
            if candidates == 0 {
                return false;
            }
        }
        false
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_match_plans_use_or_semantics() {
        let prefixes = vec!["sun".to_string(), "moon".to_string()];
        let prefix_plan = build_match_plan(&prefixes, true, false);
        assert!(match_plan_matches(&prefix_plan, "sunset", false));
        assert!(match_plan_matches(&prefix_plan, "moonbeam", false));
        assert!(!match_plan_matches(&prefix_plan, "star", false));

        let suffixes = vec!["mint".to_string(), "key".to_string()];
        let suffix_plan = build_match_plan(&suffixes, false, false);
        assert!(match_plan_matches(&suffix_plan, "seedmint", true));
        assert!(match_plan_matches(&suffix_plan, "vanitykey", true));
        assert!(!match_plan_matches(&suffix_plan, "seed", true));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_pattern_limit_returns_a_cli_error() {
        let prefixes = BS58_ALPHABET
            .chars()
            .take(MAX_GPU_PATTERNS + 1)
            .map(|character| character.to_string())
            .collect();
        let targets = SearchTargets::new(prefixes, Vec::new(), false);
        assert!(targets.validate_gpu_pattern_count().is_err());
    }
}
