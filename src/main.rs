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
        bpf_loader_upgradeable::{
            self, get_program_data_address, UpgradeableLoaderState,
        },
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
    array, fs,
    io::Write,
    str::FromStr,
    sync::{
        atomic::{
            AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering,
        },
        Arc, Mutex,
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

/// Repeatable vanity target: a pubkey matches if it matches any `--pattern`.
/// Grind stops once every `--pattern` has `--count` hits. Extra hits of
/// an easy kind are kept up to `--max-count` (default: `--count`); that
/// kind is then dropped from the search.
///
/// Syntax (`.` is not in base58, so `...` is an unambiguous split):
///   `--pattern Cavey...CooL`   prefix `Cavey` and suffix `CooL`
///   `--pattern Harmonic...`    prefix only
///   `--pattern ...pump`        suffix only
///   `--pattern Harmonic`       prefix only (the `...` may be omitted)
#[derive(Debug, Parser)]
pub struct PatternArgs {
    /// Vanity pattern. Repeatable; a hit matches any unfilled kind.
    #[clap(long, action = clap::ArgAction::Append)]
    pub pattern: Vec<String>,

    /// Whether user cares about the case of the pubkey
    #[clap(long, default_value_t = false)]
    pub case_insensitive: bool,
}

#[derive(Debug, Parser)]
pub struct GrindArgs {
    /// The pubkey that will be the signer for the CreateAccountWithSeed instruction
    #[clap(long, value_parser = parse_pubkey)]
    pub base: Pubkey,

    /// The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
    #[clap(long, value_parser = parse_pubkey)]
    pub owner: Pubkey,

    #[clap(flatten)]
    pub spec: PatternArgs,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,

    /// Stop once every --pattern has this many matches
    #[clap(long, default_value_t = 1)]
    pub count: u32,

    /// Keep collecting each --pattern up to this many while waiting
    /// for --count. Defaults to --count.
    #[clap(long)]
    pub max_count: Option<u32>,
}

#[derive(Debug, Parser)]
pub struct GrindKeypairArgs {
    #[clap(flatten)]
    pub spec: PatternArgs,

    /// Number of gpus to use for mining
    #[clap(long, default_value_t = 1)]
    #[cfg(feature = "gpu")]
    pub num_gpus: u32,

    /// Number of cpu threads to use for mining
    #[clap(long, default_value_t = 0)]
    pub num_cpus: u32,

    /// Stop once every --pattern has this many matches
    #[clap(long, default_value_t = 1)]
    pub count: u32,

    /// Keep collecting each --pattern up to this many while waiting
    /// for --count. Defaults to --count.
    #[clap(long)]
    pub max_count: Option<u32>,
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
    #[clap(
        long,
        default_value = "https://api.mainnet-beta.solana.com"
    )]
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

pub(crate) static FOUND_PER: [AtomicU32; fast::MAX_PATTERNS] =
    [const { AtomicU32::new(0) }; fast::MAX_PATTERNS];
pub(crate) static N_KINDS: AtomicUsize = AtomicUsize::new(1);
pub(crate) static TOTAL_ATTEMPTS: AtomicU64 = AtomicU64::new(0);
/// Set by the Ctrl-C handler; makes every grind loop wind down. Needed
/// because with no explicit handler, SIGINT relies on the kernel's default
/// terminate action — which does not apply when the process is PID 1 in a
/// container (common on GPU cloud hosts), so Ctrl-C would otherwise be ignored.
pub(crate) static ABORTED: AtomicBool = AtomicBool::new(false);

pub(crate) fn reset_quotas(n_kinds: usize) {
    assert!((1..=fast::MAX_PATTERNS).contains(&n_kinds), "kind count");
    N_KINDS.store(n_kinds, Ordering::SeqCst);
    for slot in FOUND_PER.iter() {
        slot.store(0, Ordering::SeqCst);
    }
}

/// Bits for kinds that still have room under `quota` (at most `quota` each).
pub(crate) fn unfilled_mask(quota: u32) -> u64 {
    let n = N_KINDS.load(Ordering::Relaxed);
    let mut mask = 0u64;
    for i in 0..n {
        if FOUND_PER[i].load(Ordering::Relaxed) < quota {
            mask |= 1u64 << i;
        }
    }
    mask
}

/// Credit every *unfilled* kind in `mask`. Caps each kind at `quota`.
/// Returns true if any kind was still under quota (print / save this hit).
pub(crate) fn credit_kinds(mask: u64, quota: u32) -> bool {
    let mask = mask & unfilled_mask(quota);
    if mask == 0 {
        return false;
    }
    let n = N_KINDS.load(Ordering::Relaxed);
    let mut useful = false;
    for i in 0..n {
        if mask & (1u64 << i) == 0 {
            continue;
        }
        loop {
            let prev = FOUND_PER[i].load(Ordering::SeqCst);
            if prev >= quota {
                break;
            }
            if FOUND_PER[i]
                .compare_exchange(
                    prev,
                    prev + 1,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                )
                .is_ok()
            {
                useful = true;
                break;
            }
        }
    }
    useful
}

pub(crate) fn quotas_filled(quota: u32) -> bool {
    let n = N_KINDS.load(Ordering::Relaxed);
    (0..n).all(|i| FOUND_PER[i].load(Ordering::Relaxed) >= quota)
}

pub(crate) fn start_grind(n_kinds: usize) {
    TOTAL_ATTEMPTS.store(0, Ordering::SeqCst);
    ABORTED.store(false, Ordering::SeqCst);
    reset_quotas(n_kinds);
}

fn done(target: u32) -> bool {
    quotas_filled(target) || ABORTED.load(Ordering::SeqCst)
}

/// `--count` is the stop threshold; `--max-count` is the per-kind save cap.
fn resolve_counts(count: u32, max_count: Option<u32>) -> (u32, u32) {
    let count = count.max(1);
    let max_count = max_count.unwrap_or(count);
    assert!(
        max_count >= count,
        "--max-count ({max_count}) must be >= --count ({count})"
    );
    (count, max_count)
}

fn targets_header(count: u32, max_count: u32) -> String {
    if max_count == count {
        format!("targets ({count} of each):")
    } else {
        format!("targets ({count} of each, max {max_count}):")
    }
}

// ─── bs58 probability (from cavemanloverboy/bs58p) ──────────────────────────

const BS58_ALPHABET: &str =
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

fn bs58_pure_prefix_suffix_prob(
    prefix: &str,
    suffix: &str,
    n_bytes: usize,
) -> f64 {
    if prefix.is_empty() && suffix.is_empty() {
        return 1.0;
    }
    let b58 = BigUint::from(58u32);
    let prefix_len = prefix.len();
    let suffix_len = suffix.len();

    let mut p_val = BigUint::zero();
    for (i, c) in prefix.chars().enumerate() {
        let idx = BS58_ALPHABET.find(c).unwrap();
        p_val +=
            BigUint::from(idx) * b58.pow((prefix_len - 1 - i) as u32);
    }
    let mut s_val = BigUint::zero();
    for (i, c) in suffix.chars().enumerate() {
        let idx = BS58_ALPHABET.find(c).unwrap();
        s_val +=
            BigUint::from(idx) * b58.pow((suffix_len - 1 - i) as u32);
    }

    let m_big = BigUint::one() << (8 * n_bytes);
    let ln2 = std::f64::consts::LN_2;
    let ln58 = 58f64.ln();
    let bits = (8 * n_bytes) as f64;
    let l_max = (bits * ln2 / ln58).ceil() as usize;

    let mut total = BigUint::zero();
    let modulus = b58.pow(suffix_len as u32);
    let start_l =
        std::cmp::max(std::cmp::max(prefix_len, suffix_len), 1);

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
        let cnt = BigUint::one()
            + (&high_pref - BigUint::one() - &first) / &modulus;
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

fn bs58_probability(
    prefix: &str,
    suffix: &str,
    case_insensitive: bool,
) -> f64 {
    let zeros = prefix
        .chars()
        .take_while(|&c| c == '1')
        .count();
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

#[allow(dead_code)]
fn expected_attempts(
    prefix: &str,
    suffix: &str,
    case_insensitive: bool,
) -> f64 {
    let p = bs58_probability(prefix, suffix, case_insensitive);
    if p <= 0.0 {
        f64::INFINITY
    } else {
        1.0 / p
    }
}

// ─── --pattern ──────────────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq, Eq)]
struct VanityPattern {
    prefix: String,
    suffix: String,
}

#[derive(Clone, Debug)]
struct VanitySpec {
    patterns: Vec<VanityPattern>,
    case_insensitive: bool,
}

impl VanitySpec {
    fn pairs(&self) -> Vec<(&str, &str)> {
        self.patterns
            .iter()
            .map(|p| (p.prefix.as_str(), p.suffix.as_str()))
            .collect()
    }

    fn match_mask(&self, pubkey: &str) -> u64 {
        let mut mask = 0u64;
        for (i, p) in self.patterns.iter().enumerate() {
            if matches_target(
                pubkey,
                &p.prefix,
                &p.suffix,
                self.case_insensitive,
            ) {
                mask |= 1u64 << i;
            }
        }
        mask
    }

    fn live_targets(&self, count: u32, max_count: u32) -> LiveTargets {
        LiveTargets {
            count,
            max_count,
            rows: self
                .patterns
                .iter()
                .map(|p| {
                    let probability = bs58_probability(
                        &p.prefix,
                        &p.suffix,
                        self.case_insensitive,
                    );
                    let expected = if probability <= 0.0 {
                        f64::INFINITY
                    } else {
                        count.max(1) as f64 / probability
                    };
                    TargetRow {
                        label: format_target_label(
                            &p.prefix, &p.suffix,
                        ),
                        probability,
                        expected,
                    }
                })
                .collect(),
        }
    }
}

/// Split `Prefix...Suffix`, `Prefix...`, `...Suffix`, or a bare prefix.
fn parse_pattern(raw: &str) -> VanityPattern {
    let (prefix, suffix) = match raw.split_once("...") {
        Some((p, s)) => {
            assert!(
                !s.contains("..."),
                "--pattern may contain only one '...' separator"
            );
            (p.to_string(), s.to_string())
        }
        None => (raw.to_string(), String::new()),
    };
    assert!(
        !prefix.is_empty() || !suffix.is_empty(),
        "--pattern must include a prefix and/or suffix (e.g. Harmonic... or ...CooL or Cavey...CooL)"
    );
    assert!(
        prefix.len() <= fast::MAX_PATTERN_LEN
            && suffix.len() <= fast::MAX_PATTERN_LEN,
        "--pattern prefix/suffix each at most {} characters",
        fast::MAX_PATTERN_LEN
    );
    VanityPattern { prefix, suffix }
}

fn resolve_spec(args: &PatternArgs) -> VanitySpec {
    let raw = &args.pattern;
    assert!(!raw.is_empty(), "supply at least one --pattern");
    assert!(
        raw.len() <= fast::MAX_PATTERNS,
        "at most {} --pattern flags",
        fast::MAX_PATTERNS
    );

    let mut patterns = Vec::with_capacity(raw.len());
    for r in raw {
        let pat = parse_pattern(r);
        validate_bs58("pattern prefix", &pat.prefix);
        validate_bs58("pattern suffix", &pat.suffix);
        patterns.push(VanityPattern {
            prefix: maybe_bs58_aware_lowercase(
                &pat.prefix,
                args.case_insensitive,
            ),
            suffix: maybe_bs58_aware_lowercase(
                &pat.suffix,
                args.case_insensitive,
            ),
        });
    }
    VanitySpec {
        patterns,
        case_insensitive: args.case_insensitive,
    }
}

// ─── formatting ─────────────────────────────────────────────────────────────

struct TargetRow {
    label: String,
    probability: f64,
    expected: f64,
}

struct LiveTargets {
    count: u32,
    max_count: u32,
    rows: Vec<TargetRow>,
}

struct UiState {
    live: Option<LiveTargets>,
    start: Option<Instant>,
    painted: bool,
}

static UI: Mutex<UiState> = Mutex::new(UiState {
    live: None,
    start: None,
    painted: false,
});

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

fn format_expected_attempts(expected: f64) -> String {
    if expected.is_finite() {
        (expected as u64).to_formatted_string(&Locale::en)
    } else {
        "∞".into()
    }
}

fn format_eta(
    remaining_hits: u32,
    probability: f64,
    rate: f64,
) -> String {
    if remaining_hits == 0 {
        return "done".into();
    }
    if rate <= 0.0 || probability <= 0.0 {
        return "—".into();
    }
    format_duration((remaining_hits as f64 / probability) / rate)
}

fn format_kind_eta(
    found: u32,
    count: u32,
    max_count: u32,
    probability: f64,
    rate: f64,
) -> String {
    if found >= max_count {
        return "capped".into();
    }
    if found >= count {
        return "collecting".into();
    }
    format_eta(count.saturating_sub(found), probability, rate)
}

fn format_target_live_line(
    row: &TargetRow,
    found: u32,
    count: u32,
    max_count: u32,
    rate: f64,
) -> String {
    format!(
        "    {}  {}/{} | {:.6e} | expected {} | ETA {}",
        row.label,
        found,
        max_count,
        row.probability,
        format_expected_attempts(row.expected),
        format_kind_eta(found, count, max_count, row.probability, rate),
    )
}

fn format_stats_line(total: u64, rate: f64, elapsed: f64) -> String {
    format!(
        "{} attempts | {} attempts/sec | elapsed: {}",
        total.to_formatted_string(&Locale::en),
        (rate as u64).to_formatted_string(&Locale::en),
        format_duration(elapsed),
    )
}

fn ui_rates(start: Instant) -> (u64, f64, f64) {
    let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    (total, total as f64 / elapsed, elapsed)
}

/// Cursor stays on the stats line (no trailing newline) after a paint.
fn paint_live(st: &mut UiState, first: bool) {
    let Some(ref live) = st.live else {
        return;
    };
    let Some(start) = st.start else {
        return;
    };
    let (total, rate, elapsed) = ui_rates(start);
    let t = live.rows.len();
    if !first && st.painted && t > 0 {
        eprint!("\x1b[{t}A");
    }
    for (i, row) in live.rows.iter().enumerate() {
        let found = FOUND_PER[i].load(Ordering::Relaxed);
        eprint!(
            "\r\x1b[K{}\n",
            format_target_live_line(
                row,
                found,
                live.count,
                live.max_count,
                rate,
            )
        );
    }
    eprint!("\r\x1b[K{}", format_stats_line(total, rate, elapsed));
    let _ = std::io::stderr().flush();
    st.painted = true;
}

fn clear_live(st: &mut UiState) {
    if !st.painted {
        return;
    }
    let t = st
        .live
        .as_ref()
        .map(|l| l.rows.len())
        .unwrap_or(0);
    if t > 0 {
        eprint!("\x1b[{t}A");
    }
    eprint!("\r\x1b[J");
    let _ = std::io::stderr().flush();
    st.painted = false;
}

fn ui_start(live: LiveTargets, start: Instant) {
    let mut st = UI.lock().unwrap();
    st.live = Some(live);
    st.start = Some(start);
    st.painted = false;
    paint_live(&mut st, true);
}

fn ui_refresh() {
    let mut st = UI.lock().unwrap();
    if st.live.is_none() {
        return;
    }
    paint_live(&mut st, false);
}

pub(crate) fn ui_with_match(f: impl FnOnce()) {
    let mut st = UI.lock().unwrap();
    clear_live(&mut st);
    f();
    paint_live(&mut st, true);
}

fn ui_stop() {
    let mut st = UI.lock().unwrap();
    if st.painted {
        eprintln!();
        st.painted = false;
    }
    st.live = None;
}

fn spawn_ui_reporter(
    shutdown: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        thread::sleep(Duration::from_secs(1));
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        ui_refresh();
    })
}

fn print_status(total: u64, rate: f64, elapsed: f64, expected: f64) {
    let e_time = if rate > 0.0 && expected.is_finite() {
        format!(
            " | E[grind_time] = {}",
            format_duration(expected / rate)
        )
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

// ─── gpu_worker_loop ────────────────────────────────────────────────────────────────

#[cfg(feature = "gpu")]
use std::ffi::c_void;

#[cfg(feature = "gpu")]
struct GpuOps {
    launch: unsafe extern "C" fn(*mut c_void, *const u8),
    query: unsafe extern "C" fn(*mut c_void) -> i32,
    read: unsafe extern "C" fn(*mut c_void, *mut u8),
    destroy: unsafe extern "C" fn(*mut c_void),
    /// Drop filled kinds from the device matcher. None for doppler.
    set_active_mask: Option<unsafe extern "C" fn(*mut c_void, u64)>,
}

/// Drive `num_gpus` async kernel launches until the grind is done.
///
/// `OUT` is the kernel's output buffer size; its last 8 bytes are the
/// `le` attempt count. `on_result(gpu, payload, secs)` gets the full `OUT`
/// bytes and decides whether it's a match.
#[cfg(feature = "gpu")]
fn run_gpu_workers<const OUT: usize>(
    num_gpus: u32,
    ops: GpuOps,
    init: impl Fn(u32) -> *mut c_void,
    done: impl Fn() -> bool,
    mut on_result: impl FnMut(usize, &[u8], f64),
    quota: u32,
) {
    let contexts: Vec<*mut c_void> = (0..num_gpus).map(&init).collect();
    let mut launch_times = vec![Instant::now(); num_gpus as usize];
    let mut in_flight = vec![false; num_gpus as usize];
    let push_mask = |ctx: *mut c_void| {
        let Some(set) = ops.set_active_mask else {
            return;
        };
        unsafe { set(ctx, unfilled_mask(quota)) };
    };
    for &ctx in &contexts {
        push_mask(ctx);
    }

    let read = |ctx: *mut c_void| -> ([u8; OUT], u64) {
        let mut out = [0u8; OUT];
        unsafe { (ops.read)(ctx, out.as_mut_ptr()) };
        let count =
            u64::from_le_bytes(array::from_fn(|j| out[OUT - 8 + j]));
        (out, count)
    };

    for (i, &ctx) in contexts.iter().enumerate() {
        let seed = new_gpu_seed(i as u32);
        launch_times[i] = Instant::now();
        unsafe { (ops.launch)(ctx, seed.as_ptr()) };
        in_flight[i] = true;
    }

    loop {
        if done() {
            break;
        }
        let mut any_ready = false;

        for (i, &ctx) in contexts.iter().enumerate() {
            if !in_flight[i] || unsafe { (ops.query)(ctx) } == 0 {
                continue;
            }
            any_ready = true;

            let secs = launch_times[i].elapsed().as_secs_f64();
            let (out, count) = read(ctx);
            TOTAL_ATTEMPTS.fetch_add(count, Ordering::Relaxed);
            on_result(i, &out, secs);

            in_flight[i] = false;
            if !done() {
                push_mask(ctx);
                let seed = new_gpu_seed(i as u32);
                launch_times[i] = Instant::now();
                unsafe { (ops.launch)(ctx, seed.as_ptr()) };
                in_flight[i] = true;
            }
        }
        if !any_ready {
            thread::sleep(Duration::from_millis(10));
        }
    }

    for (i, &ctx) in contexts.iter().enumerate() {
        if !in_flight[i] {
            continue;
        }
        while unsafe { (ops.query)(ctx) } == 0 {
            thread::sleep(Duration::from_millis(10));
        }
        TOTAL_ATTEMPTS.fetch_add(read(ctx).1, Ordering::Relaxed);
    }
    for ctx in contexts {
        unsafe { (ops.destroy)(ctx) };
    }
}
// ─── main ───────────────────────────────────────────────────────────────────

fn main() {
    rayon::ThreadPoolBuilder::new()
        .build_global()
        .unwrap();

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
    let result =
        Pubkey::create_with_seed(&base, &seed, &owner).unwrap();
    println!("Results:");
    println!("  base  {base}");
    println!("  owner {owner}");
    println!("  seed  {seed}\n");
    println!("  resulting pubkey: {result}")
}

#[cfg(feature = "deploy")]
fn deploy(args: DeployArgs) {
    let base_keypair = read_keypair_file(&args.base)
        .expect("failed to read base keypair");
    let payer_keypair = args
        .payer
        .as_ref()
        .map(|payer| {
            read_keypair_file(payer)
                .expect("failed to read payer keypair")
        })
        .unwrap_or(base_keypair.insecure_clone());
    let authority = args
        .authority
        .unwrap_or_else(|| payer_keypair.pubkey());

    let target = Pubkey::create_with_seed(
        &base_keypair.pubkey(),
        &args.seed,
        &args.owner,
    )
    .unwrap();
    let rpc_client = RpcClient::new(args.rpc);
    let buffer_len = rpc_client
        .get_account_data(&args.buffer)
        .unwrap()
        .len();
    let rent = rpc_client
        .get_minimum_balance_for_rent_exemption(
            UpgradeableLoaderState::size_of_program(),
        )
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
    let blockhash = rpc_client
        .get_latest_blockhash()
        .unwrap();
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
            &UpgradeableLoaderInstruction::DeployWithMaxDataLen {
                max_data_len,
            },
            vec![
                AccountMeta::new(*payer_address, true),
                AccountMeta::new(programdata_address, false),
                AccountMeta::new(*program_address, false),
                AccountMeta::new(*buffer_address, false),
                AccountMeta::new_readonly(sysvar::rent::id(), false),
                AccountMeta::new_readonly(sysvar::clock::id(), false),
                AccountMeta::new_readonly(system_program::id(), false),
                AccountMeta::new_readonly(
                    *upgrade_authority_address,
                    true,
                ),
            ],
        ),
    ]
}

// ─── grind ──────────────────────────────────────────────────────────────────

fn grind(mut args: GrindArgs) {
    maybe_update_num_cpus(&mut args.num_cpus);
    let spec = resolve_spec(&args.spec);
    let (target_count, max_count) =
        resolve_counts(args.count, args.max_count);
    start_grind(spec.patterns.len());
    let live = spec.live_targets(target_count, max_count);
    #[cfg(feature = "gpu")]
    eprintln!("using {} cpus, {} gpus", args.num_cpus, args.num_gpus);
    #[cfg(not(feature = "gpu"))]
    eprintln!("using {} cpus", args.num_cpus);
    eprintln!("{}", targets_header(target_count, max_count));

    let shutdown = Arc::new(AtomicBool::new(false));
    let grind_start = Instant::now();
    ui_start(live, grind_start);
    let spec = Arc::new(spec);

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let base = args.base;
        let owner = args.owner;
        let pairs = spec.pairs();
        let blob = Arc::new(
            fast::MatchTargets::new(&pairs, spec.case_insensitive)
                .gpu_blob(),
        );
        let spec = Arc::clone(&spec);
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    run_gpu_workers::<24>(
                        num_gpus,
                        GpuOps {
                            launch: gpu_grind_launch,
                            query: gpu_grind_query,
                            read: gpu_grind_read,
                            destroy: gpu_grind_destroy,
                            set_active_mask: Some(
                                gpu_grind_set_active_mask,
                            ),
                        },
                        |id| unsafe {
                            gpu_grind_init(
                                id as i32,
                                base.as_ref().as_ptr(),
                                owner.as_ref().as_ptr(),
                                blob.as_ptr(),
                                blob.len() as u64,
                                spec.case_insensitive,
                            )
                        },
                        || done(target_count),
                        |i, out, time_sec| {
                            let reconstructed: [u8; 32] = Sha256::new()
                                .chain_update(base)
                                .chain_update(&out[..16])
                                .chain_update(owner)
                                .finalize()
                                .into();
                            let out_str =
                                fd_bs58::encode_32(reconstructed);

                            let mask = spec.match_mask(&out_str);
                            if mask != 0
                                && credit_kinds(mask, max_count)
                            {
                                ui_with_match(|| {
                                    eprintln!(
                                        "gpu {} match: {} in {:.3}s",
                                        i, out_str, time_sec
                                    );
                                    eprintln!(
                                        "out seed = {out:?} -> {}",
                                        core::str::from_utf8(
                                            &out[..16]
                                        )
                                        .unwrap()
                                    );
                                });
                            }
                        },
                        max_count,
                    )
                })
                .unwrap(),
        )
    } else {
        None
    };

    let reporter = spawn_ui_reporter(Arc::clone(&shutdown));

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

            let mask = spec.match_mask(&pubkey);
            if mask != 0 && credit_kinds(mask, max_count) {
                if local_batch > 0 {
                    TOTAL_ATTEMPTS.fetch_add(local_batch, Ordering::Relaxed);
                    local_batch = 0;
                }
                let time_secs = timer.elapsed().as_secs_f64();
                ui_with_match(|| {
                    eprintln!(
                        "cpu {i} match: {pubkey}; {seed:?} -> {} in {:.3}s",
                        core::str::from_utf8(&seed).unwrap(),
                        time_secs,
                    );
                });
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
    ui_stop();

    let total = TOTAL_ATTEMPTS.load(Ordering::Relaxed);
    let elapsed = grind_start
        .elapsed()
        .as_secs_f64()
        .max(1e-9);
    let rate = total as f64 / elapsed;
    eprintln!(
        "done: {} attempts in {} at {} attempts/sec",
        total.to_formatted_string(&Locale::en),
        format_duration(elapsed),
        (rate as u64).to_formatted_string(&Locale::en)
    );
}

// ─── grind-keypair ──────────────────────────────────────────────────────────

fn grind_keypair(mut args: GrindKeypairArgs) {
    check_write_permissions();
    maybe_update_num_cpus(&mut args.num_cpus);
    let spec = resolve_spec(&args.spec);
    let (target_count, max_count) =
        resolve_counts(args.count, args.max_count);
    fast::reset_grind(spec.patterns.len());
    let live = spec.live_targets(target_count, max_count);
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
    eprintln!("{}", targets_header(target_count, max_count));
    let shutdown = Arc::new(AtomicBool::new(false));
    let grind_start = Instant::now();
    ui_start(live, grind_start);
    let match_targets = {
        let pairs = spec.pairs();
        fast::MatchTargets::new(&pairs, spec.case_insensitive)
    };
    #[cfg(feature = "gpu")]
    let spec = Arc::new(spec);
    #[cfg(not(feature = "gpu"))]
    let _ = spec;

    let reporter = spawn_ui_reporter(Arc::clone(&shutdown));

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let blob = Arc::new(match_targets.gpu_blob());
        let spec = Arc::clone(&spec);
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    run_gpu_workers::<40>(
                        num_gpus,
                        GpuOps {
                            launch: gpu_keypair_launch,
                            query: gpu_keypair_query,
                            read: gpu_keypair_read,
                            destroy: gpu_keypair_destroy,
                            set_active_mask: Some(
                                gpu_keypair_set_active_mask,
                            ),
                        },
                        |id| unsafe {
                            gpu_keypair_init(
                                id as i32,
                                blob.as_ptr(),
                                blob.len() as u64,
                                spec.case_insensitive,
                            )
                        },
                        || {
                            fast::is_done(target_count)
                                || ABORTED.load(Ordering::Relaxed)
                        },
                        |i, out, time_sec| {
                            let found_seed: [u8; 32] =
                                out[..32].try_into().unwrap();
                            let signing_key =
                                SigningKey::from_bytes(&found_seed);
                            let pubkey_bytes =
                                signing_key.verifying_key().to_bytes();
                            let pubkey_str =
                                fd_bs58::encode_32(pubkey_bytes);

                            let mask = spec.match_mask(&pubkey_str);
                            if mask != 0
                                && credit_kinds(mask, max_count)
                            {
                                ui_with_match(|| {
                                    eprintln!(
                                        "gpu {} match: {} in {:.3}s",
                                        i, pubkey_str, time_sec
                                    );
                                    eprintln!("pubkey: {pubkey_str}");
                                    save_keypair(
                                        &found_seed,
                                        &pubkey_bytes,
                                        &pubkey_str,
                                    );
                                });
                            }
                        },
                        max_count,
                    )
                })
                .unwrap(),
        )
    } else {
        None
    };

    fast::run_cpu_workers(
        &match_targets,
        args.num_cpus,
        target_count,
        max_count,
    );

    // CPU workers finished (found enough or aborted); stop GPU too.
    fast::request_abort();

    #[cfg(feature = "gpu")]
    if let Some(t) = gpu_thread {
        t.join().unwrap();
    }

    shutdown.store(true, Ordering::SeqCst);
    reporter.join().unwrap();
    ui_stop();

    let total = fast::total_attempts();
    let elapsed = grind_start
        .elapsed()
        .as_secs_f64()
        .max(1e-9);
    let rate = total as f64 / elapsed;
    eprintln!(
        "done: {} attempts in {} at {} attempts/sec",
        total.to_formatted_string(&Locale::en),
        format_duration(elapsed),
        (rate as u64).to_formatted_string(&Locale::en)
    );
}

// ─── doppler ──────────────────────────────────────────────────────────────

fn grind_doppler(mut args: DopplerArgs) {
    check_write_permissions();
    maybe_update_num_cpus(&mut args.num_cpus);
    assert!(
        (1..=4).contains(&args.segments),
        "--segments must be between 1 and 4"
    );

    let target_count = args.count;
    start_grind(1);
    let prob = doppler_probability(args.segments);
    let expected = if prob > 0.0 {
        target_count as f64 / prob
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

    let shutdown = Arc::new(AtomicBool::new(false));

    #[cfg(feature = "gpu")]
    let gpu_thread = if args.num_gpus > 0 {
        let num_gpus = args.num_gpus;
        let segments = args.segments as u32;
        Some(
            thread::Builder::new()
                .name("gpu_mgr".into())
                .spawn(move || {
                    run_gpu_workers::<40>(
                        num_gpus,
                        GpuOps {
                            launch: gpu_doppler_launch,
                            query: gpu_doppler_query,
                            read: gpu_doppler_read,
                            destroy: gpu_doppler_destroy,
                            set_active_mask: None,
                        },
                        |id| unsafe { gpu_doppler_init(id as i32, segments) },
                        || done(target_count),
                        |i, out, time_sec| {
                            let found_seed: [u8; 32] =
                                out[..32].try_into().unwrap();
                            let signing_key =
                                SigningKey::from_bytes(&found_seed);
                            let pubkey_bytes =
                                signing_key.verifying_key().to_bytes();

                            if doppler_count_segments(&pubkey_bytes)
                                >= segments
                                && credit_kinds(1, target_count)
                            {
                                let pubkey_str =
                                    fd_bs58::encode_32(pubkey_bytes);
                                eprintln!(
                                    "\r\x1b[Kgpu {} match: {} in {:.3}s",
                                    i, pubkey_str, time_sec
                                );
                                print_doppler_result(
                                    &found_seed,
                                    &pubkey_str,
                                );
                                save_keypair(
                                    &found_seed,
                                    &pubkey_bytes,
                                    &pubkey_str,
                                );
                            }
                        },
                        target_count,
                    )
                })
                .unwrap(),
        )
    } else {
        None
    };

    let grind_start = Instant::now();
    let reporter = spawn_hashrate_reporter(
        Arc::clone(&shutdown),
        expected,
        grind_start,
    );

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

            if doppler_count_segments(&pubkey_bytes) >= segments
                && credit_kinds(1, target_count)
            {
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
                print_doppler_result(&seed, &pubkey_str);
                save_keypair(&seed, &pubkey_bytes, &pubkey_str);
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
    let elapsed = grind_start
        .elapsed()
        .as_secs_f64()
        .max(1e-9);
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
        total += BINOM4[k as usize]
            * p.powi(k as i32)
            * q.powi((4 - k) as i32);
    }
    total
}

/// Print the matched keypair plus a per-segment breakdown, including the
/// assembly `.equ` constants the doppler-keygen reference emits.
fn print_doppler_result(pubkey: &[u8; 32], pubkey_str: &str) {
    eprintln!("pubkey: {pubkey_str}");
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
            let imm = i32::from_le_bytes([
                pubkey[o],
                pubkey[o + 1],
                pubkey[o + 2],
                pubkey[o + 3],
            ]);
            eprintln!(
                "  seg {s} (bytes {}-{}): imm32 {} (0x{:08x})  =>  .equ EXPECTED_KEY_{s}, 0x{:08x}",
                o,
                o + 7,
                imm,
                imm as u32,
                imm as u32
            );
        } else {
            let full =
                u64::from_le_bytes(array::from_fn(|j| pubkey[o + j]));
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

fn format_target_label(prefix: &str, suffix: &str) -> String {
    match (prefix.is_empty(), suffix.is_empty()) {
        (false, false) => format!("{}...{}", prefix, suffix),
        (false, true) => prefix.to_string(),
        (true, false) => format!("...{}", suffix),
        (true, true) => "*".to_string(),
    }
}

pub(crate) fn save_keypair(
    seed: &[u8; 32],
    pubkey: &[u8; 32],
    pubkey_str: &str,
) {
    let bytes: Vec<u8> = seed
        .iter()
        .chain(pubkey.iter())
        .copied()
        .collect();

    let path = format!("{pubkey_str}.json");
    let json = format!("{bytes:?}");

    // Create file with mode 0600 so we don't expose the private key
    // to other users on the box.
    let mut opts = fs::OpenOptions::new();
    opts.write(true)
        .create(true)
        .truncate(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }

    match opts
        .open(&path)
        .and_then(|mut f| f.write_all(json.as_bytes()))
    {
        Ok(_) => eprintln!("keypair generated at: ./{}", path),
        Err(err) => {
            eprintln!("failed to write keypair to {path}: {err}")
        }
    }
}

fn check_write_permissions() {
    let current_dir =
        std::env::current_dir().expect("we should've a dir");
    let md = fs::metadata(current_dir).unwrap();
    let permissions = md.permissions();
    let readonly = permissions.readonly();
    if readonly {
        panic!("You're trying to grind a keypair but there's no write permissions on the current dir");
    }
}

fn validate_bs58(label: &str, value: &str) {
    const BS58_CHARS: &str =
        "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    for c in value.chars() {
        assert!(
            BS58_CHARS.contains(c),
            "your {label} contains invalid bs58: {c}"
        );
    }
}

fn maybe_bs58_aware_lowercase(
    target: &str,
    case_insensitive: bool,
) -> String {
    if case_insensitive {
        target
            .chars()
            .map(|c| if c == 'L' { c } else { c.to_ascii_lowercase() })
            .collect::<String>()
    } else {
        target.to_string()
    }
}

fn matches_target(
    pubkey: &str,
    prefix: &str,
    suffix: &str,
    case_insensitive: bool,
) -> bool {
    if case_insensitive {
        (prefix.is_empty() || bs58_ci_matches(pubkey, prefix, true))
            && (suffix.is_empty()
                || bs58_ci_matches(pubkey, suffix, false))
    } else {
        pubkey.starts_with(prefix) && pubkey.ends_with(suffix)
    }
}

fn bs58_ci_matches(
    haystack: &str,
    pattern: &str,
    prefix: bool,
) -> bool {
    let h = if prefix {
        &haystack[..pattern.len().min(haystack.len())]
    } else {
        let start = haystack
            .len()
            .saturating_sub(pattern.len());
        &haystack[start..]
    };
    if h.len() != pattern.len() {
        return false;
    }
    h.bytes()
        .zip(pattern.bytes())
        .all(|(a, b)| {
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
        patterns: *const u8,
        patterns_len: u64,
        case_insensitive: bool,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_grind_launch(
        ctx: *mut std::ffi::c_void,
        seed: *const u8,
    );
    pub fn gpu_grind_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_grind_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_grind_destroy(ctx: *mut std::ffi::c_void);
    pub fn gpu_grind_set_active_mask(
        ctx: *mut std::ffi::c_void,
        mask: u64,
    );

    pub fn gpu_keypair_init(
        id: i32,
        patterns: *const u8,
        patterns_len: u64,
        case_insensitive: bool,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_keypair_launch(
        ctx: *mut std::ffi::c_void,
        seed: *const u8,
    );
    pub fn gpu_keypair_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_keypair_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_keypair_destroy(ctx: *mut std::ffi::c_void);
    pub fn gpu_keypair_set_active_mask(
        ctx: *mut std::ffi::c_void,
        mask: u64,
    );

    pub fn gpu_doppler_init(
        id: i32,
        required_segments: u32,
    ) -> *mut std::ffi::c_void;
    pub fn gpu_doppler_launch(
        ctx: *mut std::ffi::c_void,
        seed: *const u8,
    );
    pub fn gpu_doppler_query(ctx: *mut std::ffi::c_void) -> i32;
    pub fn gpu_doppler_read(ctx: *mut std::ffi::c_void, out: *mut u8);
    pub fn gpu_doppler_destroy(ctx: *mut std::ffi::c_void);
}

#[cfg(feature = "gpu")]
fn new_gpu_seed(gpu_id: u32) -> [u8; 32] {
    Sha256::new()
        .chain_update(rand::random::<[u8; 32]>())
        .chain_update(gpu_id.to_le_bytes())
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
    fn parse_pattern_splits_on_ellipsis() {
        assert_eq!(
            parse_pattern("Cavey...CooL"),
            VanityPattern {
                prefix: "Cavey".into(),
                suffix: "CooL".into(),
            }
        );
        assert_eq!(
            parse_pattern("Harmonic..."),
            VanityPattern {
                prefix: "Harmonic".into(),
                suffix: String::new(),
            }
        );
        assert_eq!(
            parse_pattern("...pump"),
            VanityPattern {
                prefix: String::new(),
                suffix: "pump".into(),
            }
        );
        assert_eq!(
            parse_pattern("Harmonic"),
            VanityPattern {
                prefix: "Harmonic".into(),
                suffix: String::new(),
            }
        );
    }

    #[test]
    fn match_mask_sets_bit_per_pattern() {
        let spec = VanitySpec {
            patterns: vec![
                VanityPattern {
                    prefix: "XkC".into(),
                    suffix: String::new(),
                },
                VanityPattern {
                    prefix: "zzz".into(),
                    suffix: String::new(),
                },
            ],
            case_insensitive: false,
        };
        let key = "XkCriyrNwS3G4rzAXtG5B1nnvb5Ka1JtCku93VqeKAr";
        assert_eq!(spec.match_mask(key), 0b01);
        assert_eq!(spec.match_mask("zzzABC"), 0b10);
        assert_eq!(spec.match_mask("nope"), 0);
    }
}
