# `vanity`

A _bLaZinGlY fAsT_ tool for grinding vanity addresses on Solana.

> **Note**: This project (+ the readme) was generated with the assistance of AI (Claude).

## Features

- GPU and CPU support for address grinding
- Case-insensitive matching
- Leet speak matching (e.g., 'a'='4', 'e'='3', 't'='7', etc.)
- Prefix, suffix, and "anywhere in address" matching
- Automatic GPU compute capability detection
- Output logging and key saving

## Installation

### From Cargo

```bash
# CPU-only version
cargo install vanity

# With GPU support (recommended)
cargo install vanity --features=gpu
```

### Using Docker with GPU

```bash
# Build the image
docker build -f Dockerfile.gpu -t vanity .

# Run with GPU access
docker run --gpus all vanity --help
```

## Usage

The tool has two main commands: `grind` and `deploy`.

### Grinding Addresses

```bash
vanity grind [OPTIONS] --base <BASE> --owner <OWNER>

Options:
  --base <BASE>                The pubkey that will be the signer for CreateAccountWithSeed
  --owner <OWNER>              The account owner (e.g., BPFLoaderUpgradeab1e11111111111111111111111)
  --prefix <PREFIX>            The prefix for the pubkey [default: ""]
  --suffix <SUFFIX>            The suffix for the pubkey [default: ""]
  --any <ANY>                  Search for this string anywhere in the address [default: ""]
  --case-insensitive          Whether to ignore case when matching
  --leet-speak                Enable leet speak matching (e.g., a=4, e=3, etc.)
  --logfile <LOGFILE>         Optional log file
  --num-gpus <NUM_GPUS>       Number of GPUs to use [default: 1]
  --num-cpus <NUM_CPUS>       Number of CPU threads to use [default: 0]
```

### Deploying Programs

```bash
vanity deploy [OPTIONS] --base <BASE> --owner <OWNER> --buffer <BUFFER> --seed <SEED>

Options:
  --base <BASE>                Path to base keypair file
  --rpc <RPC>                 RPC URL [default: "https://api.mainnet-beta.solana.com"]
  --owner <OWNER>             Program owner (usually BPFLoaderUpgradeab1e11111111111111111111111)
  --buffer <BUFFER>           Buffer address containing program data
  --payer <PAYER>             Optional separate payer keypair
  --seed <SEED>               Seed found during grinding
  --authority <AUTHORITY>     Program authority (defaults to payer)
  --compute-unit-price <CU>   Optional compute unit price
  --logfile <LOGFILE>         Optional log file
```

## Examples

### Finding a Vanity Address

```bash
# Search for an address starting with "COOL"
vanity grind --base <PUBKEY> --owner <OWNER> --prefix COOL

# Search for an address ending with "NICE"
vanity grind --base <PUBKEY> --owner <OWNER> --suffix NICE

# Search for "RUST" anywhere in the address
vanity grind --base <PUBKEY> --owner <OWNER> --any RUST

# Case-insensitive search with leet speak
vanity grind --base <PUBKEY> --owner <OWNER> --any ELITE --case-insensitive --leet-speak
```

### Deploying with Found Seed

```bash
vanity deploy --base keypair.json --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --buffer <BUFFER_PUBKEY> --seed <FOUND_SEED>
```

## Performance

Performance varies by hardware:

- RTX 4090: ~1 billion addresses/second
- RTX 3050: ~300 million addresses/second

The tool automatically detects your GPU's compute capability and optimizes accordingly.

## Leet Speak Transformations

The following character transformations are supported:

- a/A ↔ 4
- e/E ↔ 3
- t/T ↔ 7
- l/L/i/I ↔ 1
- s/S ↔ 5
- g/G ↔ 6
- b/B ↔ 8
- z/Z ↔ 2

## Acknowledgements

- SHA256 implementation from [cuda-hashing-algos](https://github.com/mochimodev/cuda-hashing-algos) (public domain)
- Base58 encoding adapted from firedancer (APACHE-2.0)
