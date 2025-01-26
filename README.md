# `vanity`

A _bLaZinGlY fAsT_ tool for grinding vanity addresses on Solana.

> **Note**: This project (+ the readme) was generated with the assistance of AI (Claude) in Cursor.

## Features

### Core Features

- Multi-GPU support with automatic device detection and optimization
- Multi-threaded CPU support with automatic thread count detection
- Automatic compute capability detection for optimal GPU performance
- Base58-aware case-insensitive matching
- Smart leet speak transformations with bidirectional matching
- Flexible search patterns:
  - Prefix matching (start of address)
  - Suffix matching (end of address)
  - Anywhere matching (substring in address)
  - Combinations of all three patterns

### Search Capabilities

- Case-sensitive and case-insensitive matching
- Bidirectional leet speak transformations (e.g., both 'a' → '4' and '4' → 'a')
- Smart character mapping for invalid Base58 characters:
  - '0' → 'o'
  - 'I' → '1'
  - 'O' → 'o'
  - 'l' → 'L'
- Automatic validation of search strings
- Real-time performance metrics and progress reporting

### Output and Logging

- Detailed logging with configurable log files
- Performance statistics:
  - Iterations per second
  - Time to find matches
  - GPU/CPU utilization
- Automatic key saving:
  - Saves found addresses and seeds
  - Human-readable format
  - Organized file structure
- Validation mismatch detection and debugging

### Deployment Support

- Direct program deployment with found seeds
- Flexible deployment options:
  - Custom RPC endpoints
  - Separate payer accounts
  - Configurable program authority
  - Compute unit price adjustment
- Support for BPF Loader Upgradeable programs

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
  --ci                        Whether to ignore case when matching
  --leet                      Enable leet speak matching (e.g., a=4, e=3, etc.)
  --logfile <LOGFILE>         Optional log file
  --num-cpus <NUM_CPUS>       Number of CPU threads to use [default: 0]
```

### Deploying Programs

```bash
vanity deploy [OPTIONS] --base <BASE> --owner <OWNER> --buffer <BUFFER> --seed <SEED>

Options:
  --base <BASE>               Path to base keypair file
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
vanity grind --base <PUBKEY> --owner <OWNER> --any ELITE --case-insensitive --leet

# Combined search patterns
vanity grind --base <PUBKEY> --owner <OWNER> --prefix COOL --suffix NICE --any RUST

# With logging enabled
vanity grind --base <PUBKEY> --owner <OWNER> --prefix TEST --logfile output.log
```

### Deploying with Found Seed

```bash
# Basic deployment
vanity deploy --base keypair.json --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --buffer <BUFFER_PUBKEY> --seed <FOUND_SEED>

# Deployment with custom RPC and payer
vanity deploy --base keypair.json --owner BPFLoaderUpgradeab1e11111111111111111111111 \
  --buffer <BUFFER_PUBKEY> --seed <FOUND_SEED> \
  --rpc "https://api.devnet.solana.com" \
  --payer payer_keypair.json \
  --compute-unit-price 1000
```

## Technical Details

### Base58 Character Set

The tool uses the standard Base58 character set:

```
123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz
```

### Character Mapping

Invalid characters are automatically mapped to valid Base58 equivalents:

- '0' → 'o'
- 'I' → '1'
- 'O' → 'o'
- 'l' → '1'

### Leet Speak Transformations

The following bidirectional character transformations are supported:

- a/A ↔ 4
- e/E ↔ 3
- t/T ↔ 7
- l/L/i/I ↔ 1
- s/S ↔ 5
- g/G ↔ 6
- b/B ↔ 8
- z/Z ↔ 2

### Performance

Performance varies by hardware. The tool automatically:

- Detects available GPUs
- Optimizes for each GPU's compute capability
- Utilizes all available GPU cores
- Adjusts CPU thread count based on system capabilities

### Output Format

Found addresses are saved in the following format:

```
<address> -> [seed_bytes] [seed_utf8]
```

## Acknowledgements

- SHA256 implementation from [cuda-hashing-algos](https://github.com/mochimodev/cuda-hashing-algos) (public domain)
- Base58 encoding adapted from firedancer (APACHE-2.0)
