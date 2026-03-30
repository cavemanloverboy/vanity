use clap::Parser;
use ed25519_dalek::SigningKey;
use num_bigint::BigUint;
use num_format::{Locale, ToFormattedString};
use num_traits::{One, ToPrimitive, Zero};
use rand;
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
    Verify(VerifyArgs),
    #[cfg(feature = "deploy")]
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

    #[clap(long)]
    pub suffix: Option<String>,

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
pub struct GrindKeypairArgs {
    /// The target prefix for the pubkey
    #[clap(long)]
    pub prefix: Option<String>,

    /// The target suffix for the pubkey
    #[clap(long)]
    pub suffix: Option<String>,

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

fn done(target: u32) -> bool {
    FOUND.load(Ordering::SeqCst) >= target
}

// ─── bs58 probability (from cavemanloverboy/bs58p) ──────────────────────────

const BS58_ALPHABET: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

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
        let ci_factor: f64 = prefix
            .chars()
            .chain(suffix.chars())
            .filter(|c| c.is_ascii_alphabetic() && *c != 'L')
            .fold(1.0, |acc, _| acc * 2.0);
        prob * ci_factor
    } else {
        prob
    }
}

fn expected_attempts(prefix: &str, suffix: &str, case_insensitive: bool) -> f64 {
    let p = bs58_probability(prefix, suffix, case_insensitive);
    if p <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / p
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
    thread::spawn(move || {
        loop {
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
        }
    })
}

// ─── main ───────────────────────────────────────────────────────────────────

fn main() {
    rayon::ThreadPoolBuilder::new().build_global().unwrap();

    let command = Command::parse();
    match command {
        Command::Grind(args) => grind(args),
        Command::GrindKeypair(args) => grind_keypair(args),
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

    let target =
        Pubkey::create_with_seed(&base_keypair.pubkey(), &args.seed, &args.owner).unwrap();
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
    let prefix = get_validated_bs58("prefix", &args.prefix, args.case_insensitive);
    let suffix = get_validated_bs58("suffix", &args.suffix, args.case_insensitive);

    let expected = expected_attempts(prefix, suffix, args.case_insensitive);
    let prob = bs58_probability(prefix, suffix, args.case_insensitive);
    #[cfg(feature = "gpu")]
    eprintln!("using {} cpus, {} gpus", args.num_cpus, args.num_gpus);
    #[cfg(not(feature = "gpu"))]
    eprintln!("using {} cpus", args.num_cpus);
    let target_label = format_target_label(prefix, suffix);
    eprintln!(
        "target: {} | probability: {:.6e} | expected: {} attempts",
        target_label, prob, (expected as u64).to_formatted_string(&Locale::en)
    );

    let target_count = args.count;
    let shutdown = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let base = args.base;
        let owner = args.owner;
        let ci = args.case_insensitive;
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    let mut contexts = Vec::with_capacity(num_gpus as usize);
                    for id in 0..num_gpus {
                        let ctx = unsafe {
                            gpu_grind_init(
                                id as i32,
                                base.as_ref().as_ptr(),
                                owner.as_ref().as_ptr(),
                                prefix.as_ptr(),
                                prefix.len() as u64,
                                suffix.as_ptr(),
                                suffix.len() as u64,
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
                        unsafe { gpu_grind_launch(ctx, seed.as_ptr()); }
                        in_flight[i] = true;
                    }

                    loop {
                        if done(target_count) { break; }

                        let mut any_ready = false;
                        for (i, &ctx) in contexts.iter().enumerate() {
                            if !in_flight[i] { continue; }
                            if unsafe { gpu_grind_query(ctx) } == 0 { continue; }
                            any_ready = true;

                            let time_sec = launch_times[i].elapsed().as_secs_f64();
                            let mut out = [0u8; 24];
                            unsafe { gpu_grind_read(ctx, out.as_mut_ptr()); }

                            let reconstructed: [u8; 32] = Sha256::new()
                                .chain_update(base)
                                .chain_update(&out[..16])
                                .chain_update(owner)
                                .finalize()
                                .into();
                            let out_str = fd_bs58::encode_32(reconstructed);
                            let out_str_check = maybe_bs58_aware_lowercase(&out_str, ci);
                            let count = u64::from_le_bytes(array::from_fn(|j| out[16 + j]));

                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);

                            if out_str_check.starts_with(prefix) && out_str_check.ends_with(suffix)
                            {
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
                                unsafe { gpu_grind_launch(ctx, seed.as_ptr()); }
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
                            unsafe { gpu_grind_read(ctx, out.as_mut_ptr()); }
                            let count = u64::from_le_bytes(array::from_fn(|j| out[16 + j]));
                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);
                        }
                    }
                    for ctx in contexts {
                        unsafe { gpu_grind_destroy(ctx); }
                    }
                })
                .unwrap(),
        )
    } else {
        None
    };

    let grind_start = Instant::now();
    let reporter = spawn_hashrate_reporter(
        Arc::clone(&shutdown), expected, grind_start,
    );

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
                const ALNUM: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
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

            if matches_target(&pubkey, prefix, suffix, args.case_insensitive)
            {
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
    let prefix = get_validated_bs58("prefix", &args.prefix, args.case_insensitive);
    let suffix = get_validated_bs58("suffix", &args.suffix, args.case_insensitive);

    let expected = expected_attempts(prefix, suffix, args.case_insensitive);
    let prob = bs58_probability(prefix, suffix, args.case_insensitive);
    #[cfg(feature = "gpu")]
    eprintln!("using {} cpus, {} gpus", args.num_cpus, args.num_gpus);
    #[cfg(not(feature = "gpu"))]
    eprintln!("using {} cpus", args.num_cpus);
    let target_label = format_target_label(prefix, suffix);
    eprintln!(
        "target: {} | probability: {:.6e} | expected: {} attempts",
        target_label, prob, (expected as u64).to_formatted_string(&Locale::en)
    );

    let target_count = args.count;
    let shutdown = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let ci = args.case_insensitive;
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    let mut contexts = Vec::with_capacity(num_gpus as usize);
                    for id in 0..num_gpus {
                        let ctx = unsafe {
                            gpu_keypair_init(
                                id as i32,
                                prefix.as_ptr(),
                                prefix.len() as u64,
                                suffix.as_ptr(),
                                suffix.len() as u64,
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
                        unsafe { gpu_keypair_launch(ctx, seed.as_ptr()); }
                        in_flight[i] = true;
                    }

                    loop {
                        if done(target_count) { break; }

                        let mut any_ready = false;
                        for (i, &ctx) in contexts.iter().enumerate() {
                            if !in_flight[i] { continue; }
                            if unsafe { gpu_keypair_query(ctx) } == 0 { continue; }
                            any_ready = true;

                            let time_sec = launch_times[i].elapsed().as_secs_f64();
                            let mut out = [0u8; 40];
                            unsafe { gpu_keypair_read(ctx, out.as_mut_ptr()); }

                            let found_seed: [u8; 32] = out[..32].try_into().unwrap();
                            let signing_key = SigningKey::from_bytes(&found_seed);
                            let pubkey_bytes = signing_key.verifying_key().to_bytes();
                            let pubkey_str = fd_bs58::encode_32(pubkey_bytes);
                            let pubkey_check = maybe_bs58_aware_lowercase(&pubkey_str, ci);
                            let count = u64::from_le_bytes(array::from_fn(|j| out[32 + j]));

                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);

                            if pubkey_check.starts_with(prefix)
                                && pubkey_check.ends_with(suffix)
                            {
                                eprintln!(
                                    "\r\x1b[Kgpu {} match: {} in {:.3}s",
                                    i, &pubkey_str, time_sec
                                );
                                print_keypair_result(&found_seed, &pubkey_bytes, &pubkey_str);
                                FOUND.fetch_add(1, Ordering::SeqCst);
                            }

                            in_flight[i] = false;
                            if !done(target_count) {
                                iterations[i] += 1;
                                let seed = new_gpu_seed(i as u32, iterations[i]);
                                launch_times[i] = Instant::now();
                                unsafe { gpu_keypair_launch(ctx, seed.as_ptr()); }
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
                            unsafe { gpu_keypair_read(ctx, out.as_mut_ptr()); }
                            let count = u64::from_le_bytes(array::from_fn(|j| out[32 + j]));
                            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);
                        }
                    }
                    for ctx in contexts {
                        unsafe { gpu_keypair_destroy(ctx); }
                    }
                })
                .unwrap(),
        )
    } else {
        None
    };

    let grind_start = Instant::now();
    let reporter = spawn_hashrate_reporter(
        Arc::clone(&shutdown), expected, grind_start,
    );

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
            let pubkey_str = fd_bs58::encode_32(pubkey_bytes);

            local_batch += 1;
            if local_batch >= 4096 {
                TOTAL_ATTEMPTS.fetch_add(4096, Ordering::Relaxed);
                local_batch -= 4096;
            }

            if matches_target(&pubkey_str, prefix, suffix, args.case_insensitive) {
                if local_batch > 0 {
                    TOTAL_ATTEMPTS.fetch_add(local_batch, Ordering::Relaxed);
                    local_batch = 0;
                }
                let time_secs = timer.elapsed().as_secs_f64();
                let elapsed_global = grind_start.elapsed().as_secs_f64().max(1e-9);
                let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
                let global_rate = total as f64 / elapsed_global;
                eprintln!(
                    "\r\x1b[Kcpu {i} match: {pubkey_str} in {:.3}s; {} attempts/sec",
                    time_secs,
                    (global_rate as u64).to_formatted_string(&Locale::en)
                );
                print_keypair_result(&seed, &pubkey_bytes, &pubkey_str);
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

fn format_target_label(prefix: &str, suffix: &str) -> String {
    match (prefix.is_empty(), suffix.is_empty()) {
        (false, false) => format!("{}...{}", prefix, suffix),
        (false, true) => prefix.to_string(),
        (true, false) => format!("...{}", suffix),
        (true, true) => "*".to_string(),
    }
}

fn print_keypair_result(seed: &[u8; 32], pubkey: &[u8; 32], pubkey_str: &str) {
    let seed_hex: String = seed.iter().map(|b| format!("{b:02x}")).collect();
    eprintln!("pubkey:   {pubkey_str}");
    eprintln!("seed hex: {seed_hex}");
    let keypair_json: Vec<u8> = seed.iter().chain(pubkey.iter()).copied().collect();
    eprintln!("keypair json (solana-compatible): {:?}", keypair_json);
}

fn get_validated_bs58(
    label: &str,
    value: &Option<String>,
    case_insensitive: bool,
) -> &'static str {
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    if let Some(ref s) = value {
        for c in s.chars() {
            assert!(
                BS58_CHARS.contains(c),
                "your {label} contains invalid bs58: {c}"
            );
        }
        let validated = maybe_bs58_aware_lowercase(s, case_insensitive);
        return validated.leak();
    }
    ""
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

fn matches_target(pubkey: &str, prefix: &str, suffix: &str, case_insensitive: bool) -> bool {
    if case_insensitive {
        (prefix.is_empty() || bs58_ci_matches(pubkey, prefix, true))
            && (suffix.is_empty() || bs58_ci_matches(pubkey, suffix, false))
    } else {
        pubkey.starts_with(prefix) && pubkey.ends_with(suffix)
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
    h.bytes()
        .zip(pattern.bytes())
        .all(|(a, b)| if b == b'L' { a == b'L' } else { a.to_ascii_lowercase() == b })
}

#[cfg(feature = "gpu")]
extern "C" {
    pub fn gpu_grind_init(
        id: i32,
        base: *const u8,
        owner: *const u8,
        target: *const u8,
        target_len: u64,
        suffix: *const u8,
        suffix_len: u64,
        case_insensitive: bool,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_grind_launch(ctx: *mut std::ffi::c_void, seed: *const u8);
    pub fn gpu_grind_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_grind_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_grind_destroy(ctx: *mut std::ffi::c_void);

    pub fn gpu_keypair_init(
        id: i32,
        prefix: *const u8,
        prefix_len: u64,
        suffix: *const u8,
        suffix_len: u64,
        case_insensitive: bool,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_keypair_launch(ctx: *mut std::ffi::c_void, seed: *const u8);
    pub fn gpu_keypair_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_keypair_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_keypair_destroy(ctx: *mut std::ffi::c_void);
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
