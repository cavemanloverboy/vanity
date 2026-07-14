# `vanity`

A *bLaZinGlY fAsT* tool for grinding vanity addresses on Solana.

## 1) What

Typically, solana developers wishing to obtain a vanity address for their program or token grind out ed25519 keypairs and sign off on a `SystemInstruction::CreateAccount` instruction. However, by using `SystemInstruction::CreateAccountWithSeed`, developers can bypass ed25519 and get extreme speedups on address searches. Although not as generic, this method covers many use cases.

## Install

By default, vanity compiles for cpu:

```bash
cargo install vanity
```

To compile for an NVIDIA gpu (CUDA) — requires the CUDA toolkit (`nvcc`) on the build machine:

```bash
cargo install vanity --features=gpu
```

By default the CUDA build emits code for several architectures (Turing through Ada) plus a PTX fallback. For a faster, targeted build set `VANITY_CUDA_ARCH` to your GPU's compute capability — e.g. `86` for an RTX 3090, `89` for an RTX 4090, `80` for an A100, `90` for an H100:

```bash
VANITY_CUDA_ARCH=86 cargo install vanity --features=gpu
```

If a launch prints `no kernel image is available for execution on the device`, the binary was built without your GPU's architecture — rebuild with the right `VANITY_CUDA_ARCH`.

On machines without CUDA (AMD / Intel / Apple GPUs, or no NVIDIA driver), build the OpenCL backend instead:

```bash
cargo install vanity --features=opencl
```

The OpenCL backend needs only an OpenCL 1.2 ICD loader and headers at build time (`-lOpenCL` on Linux, the system `OpenCL.framework` on macOS); kernels are compiled at runtime for whatever device is present. It exposes the same CLI and `--num-gpus` flag as the CUDA build. The two GPU backends are mutually exclusive — pick one feature at build time.

If you don't have a GPU, consider using [vast.ai](https://cloud.vast.ai/?ref_id=126830). Pls use this referral link so that I can keep using GPUs.

## Usage

vanity grinds three kinds of vanity output. All three share the same mining flags (`--num-cpus`, `--num-gpus`, `--count`) described in [Common options](#common-options).

| Command | Produces | How the address is derived | Speed |
| --- | --- | --- | --- |
| [`grind`](#grind-a-vanity-seed) | a **seed** for `CreateAccountWithSeed` | `base58(SHA256(base ‖ seed ‖ owner))` | fastest (SHA-256 only) |
| [`grind-keypair`](#grind-a-vanity-keypair) | an **ed25519 keypair** | `base58(ed25519_pubkey(seed))` | slower (ed25519 per attempt) |
| [`grind-doppler`](#grind-a-doppler-keypair) | an **ed25519 keypair** with sign-extendable segments | `ed25519_pubkey(seed)`, matched on bytes | slower (ed25519 per attempt) |

Run `vanity <command> --help` for the full flag list.

### Grind a vanity seed

`grind` is the fast path. Instead of searching ed25519 keypairs, it searches
*seeds* for `SystemInstruction::CreateAccountWithSeed`, whose resulting address
is `Pubkey::create_with_seed(base, seed, owner) = base58(SHA256(base ‖ seed ‖ owner))`.
No ed25519 is involved, so it's dramatically faster — but the account is
controlled by `base` (the signer), not by a standalone keypair.

You supply the `--base` pubkey (the signer for the create instruction), the
program `--owner`, and one or more `--prefix` and/or `--suffix` values to
match (base58):

```bash
vanity grind \
  --base <YOUR_WALLET_PUBKEY> \
  --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --prefix abc \
  --num-gpus 1
```

The match line reports the resulting address and the 16-character seed:

```text
gpu 0 match: abcJvoUM9mcmHewxCumNuG8sQGM3u51QB8Wi1RVmats in 0.603s
out seed = [...] -> JcZmEdyCAWJ7q9XI
```

Recompute the address from a `base`/`owner`/`seed` at any time with `verify`:

```bash
vanity verify \
  --base <YOUR_WALLET_PUBKEY> \
  --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --seed JcZmEdyCAWJ7q9XI
```

To use the seed on-chain, pass it to `create_account_with_seed` (the resulting
address is the `to_pubkey`):

```rust
pub fn create_account_with_seed(
    from_pubkey: &Pubkey,
    // this is the resulting address, obtained via Pubkey::create_with_seed
    to_pubkey: &Pubkey,
    base: &Pubkey,
    seed: &str,
    lamports: u64,
    space: u64,
    owner: &Pubkey,
) -> Instruction
```

### Grind a vanity keypair

`grind-keypair` searches genuine ed25519 keypairs whose **public key** matches a
base58 `--prefix` and/or `--suffix`. This is the classic "vanity wallet/mint"
case: the result is a standalone keypair you can sign with. It's slower than
`grind` because each attempt computes an ed25519 scalar multiplication. The CPU
path uses a batched custom ed25519 implementation (AVX-512 IFMA when available);
GPUs use the CUDA/OpenCL keypair kernels.

```bash
vanity grind-keypair \
  --prefix sun,moon,mint \
  --suffix key \
  --num-gpus 1
```

On a match it prints the public key, the seed (hex), and a Solana-compatible
keypair JSON array you can save to a file and use with the CLI / SDK:

```text
gpu 0 match: bob... in 0.4s
pubkey:   bob...xyz
seed hex: 68e36f80...
keypair json (solana-compatible): [104, 227, 111, ...]
```

Add `--case-insensitive` to match the prefix/suffix ignoring case (except `L`,
which has no lowercase form in base58).

Multiple values use OR within each flag and AND between the two flags. For
example, the command above matches a public key that starts with `sun`, `moon`,
or `mint` and ends with `key`. Values can be comma-separated or supplied by
repeating a flag; these commands are equivalent:

```bash
vanity grind-keypair --prefix sun,moon,mint --suffix key
vanity grind-keypair --prefix sun --prefix moon --prefix mint --suffix key
```

The same syntax is supported by `grind`. Existing single-value commands remain
unchanged.

CUDA and OpenCL searches accept up to 32 non-redundant values per flag. CPU-only
searches do not have this limit. Duplicate targets and targets covered by a
shorter alternative are removed automatically before the search starts.

### Grind a doppler keypair

`grind-doppler` searches ed25519 keypairs whose **public key** packs into
sign-extendable 32-bit immediates — useful in sBPF / VM contexts where a 64-bit
key segment that sign-extends from an `i32` can be compared with a cheaper
immediate instruction. (Inspired by
[doppler-keygen](https://github.com/blueshift-gg/doppler-keygen).)

The 32-byte pubkey is split into four 8-byte segments. A segment is
*sign-extendable* when its low 4 bytes form a little-endian `i32` and its high
4 bytes are that value's sign extension:

- bit 31 clear (positive `i32`): high 4 bytes are `00 00 00 00`
- bit 31 set (negative `i32`): high 4 bytes are `FF FF FF FF`

`--segments N` (1–4) requires **at least N** of the four segments to be
sign-extendable. Each segment matches with probability 2⁻³², so `--segments 1`
needs ~1 billion keypairs (seconds-to-minutes on a GPU) while `N ≥ 2` is
exponentially rarer.

```bash
# any one segment
vanity grind-doppler --segments 1 --num-gpus 1

# require two sign-extendable segments
vanity grind-doppler --segments 2 --num-gpus 1
```

On a match it prints the keypair (with Solana-compatible JSON array, same as
`grind-keypair`) plus a per-segment breakdown including the assembly `.equ`
constants for each segment:

```text
gpu 0 match: Gyqq...SrPP in 0.4s
pubkey:   Gyqq...SrPP
seed hex: a58004d0...
doppler: 1/4 sign-extendable segment(s)
  seg 0 (bytes 0-7):  0x2a768d76fb0b6fed (not sign-extendable)  =>  .equ EXPECTED_KEY_0, 0x2a768d76fb0b6fed
  seg 1 (bytes 8-15): imm32 149357322 (0x08e7030a)              =>  .equ EXPECTED_KEY_1, 0x08e7030a
  ...
```

### Common options

| Flag | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `--num-cpus <N>` | all | `0` (= all logical cores) | CPU mining threads |
| `--num-gpus <N>` | all (GPU builds only) | `1` | GPUs to mine on |
| `--count <N>` | all | `1` | stop after finding N matches |
| `--case-insensitive` | `grind`, `grind-keypair` | off | match prefix/suffix ignoring case |
| `--prefix` / `--suffix` | `grind`, `grind-keypair` | — | base58 targets to match; repeat or separate with commas; supply at least one |
| `--segments <1-4>` | `grind-doppler` | `1` | sign-extendable segments required |

To run purely on CPU (no GPU), build without a GPU feature, or pass `--num-gpus 0`.

## Contributions

yes

## Performance

Approximate single-device throughput:

| Backend | Device | seeds/s | keypairs/s |
| --- | --- | --- | --- |
| CPU    | AMD EPYC 9275F (48 threads, AVX-512 IFMA) | ~201 M | ~33 M |
| CUDA   | RTX 4090   | ~8.6B B | ~65 M  |
| OpenCL | Apple Silicon | ~315 M  | ~15 M  |

`grind` is far faster because each attempt is just a SHA-256 hash. Keypair modes (`grind-keypair`/`grind-doppler`) perform full sha512 and ed25519 scalar multiplication per attempt. Throughput scales roughly linearly with `--num-gpus`.


## Acknowledgements, External Libraries

- The sha2 implementation used in this library is taken from [here](https://github.com/mochimodev/cuda-hashing-algos), which is in the public domain.
- The base58 encoding implementation is taken from firedancer with heavy modifications for use in cuda & case insensitive encodings, licensed under APACHE-2.0
