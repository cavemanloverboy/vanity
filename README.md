# `vanity`

A *bLaZinGlY fAsT* tool for grinding vanity addresses on Solana.

## 1) What

Typically, solana developers wishing to obtain a vanity address for their program or token grind out ed25519 keypairs and sign off on a `SystemInstruction::CreateAccount` instruction. However, by using `SystemInstruction::CreateAccountWithSeed`, developers can bypass ed25519 and get extreme speedups on address searches. Although not as generic, this method covers many use cases.

## 2) H

By default, vanity compiles for cpu. Install via

```bash
cargo install vanity
```

To compile for gpu, install via

```bash
cargo install vanity --features=gpu
```
If you don't have a GPU, consider using [vast.ai](https://cloud.vast.ai/?ref_id=126830). Pls use this referral link so that I can keep using GPUs.


Refer to the help via `vanity --help` for information on usage.

```bash
Usage: vanity grind [OPTIONS] --base <BASE> --owner <OWNER>

Options:
      --base <BASE>          The pubkey that will be the signer for the CreateAccountWithSeed instruction
      --owner <OWNER>        The account owner, e.g. BPFLoaderUpgradeab1e11111111111111111111111 or TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA
      --prefix <PREFIX>      The target prefix for the pubkey
      --suffix <SUFFIX>      The target suffix for the pubkey
      --case-insensitive     Whether user cares about the case of the pubkey
      --logfile <LOGFILE>    Optional log file
      --num-cpus <NUM_CPUS>  Number of cpu threads to use for mining [default: 0]
  -h, --help                 Print help
```

To actually make use of the resulting seed, refer to the `solana_program` docs:

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
## Estimated Number of Attempts Needed for Each Prefix Length

| Prefix | Avg** Attempts (Case Sensitive) | Attempts 90% (Case Sensitive) | Avg** Attempts (Case Insensitive) | Attempts 90% (Case Insensitive) | Est Time CS*     | Est Time CI*    |
|--------|-------------------------------|-------------------------------|----------------------------------|----------------------------------|------------------|----------------|
| 1      | 58                            | 132                           | 33                               | 77                               | <1 ms            | <1 ms          |
| 2      | 3 thousand                    | 8 thousand                    | 1 thousand                       | 2 thousand                       | <1 ms            | <1 ms          |
| 3      | 195 thousand                  | 449 thousand                  | 39 thousand                      | 90 thousand                      | <1 ms            | <1 ms          |
| 4      | 11 million                    | 26 million                    | 1.3 million                      | 3 million                        | 9 ms             | 1 ms           |
| 5      | 656 million                   | 1.5 billion                   | 45 million                       | 104 million                      | 0.55 sec         | 38 ms          |
| 6      | 38 billion                    | 87 billion                    | 1.5 billion                      | 3.5 billion                      | 31.7 sec         | 1.25 sec       |
| 7      | 2.2 trillion                  | 5 trillion                    | 52.5 billion                     | 120.9 billion                    | 30.6 min         | 43.8 sec       |
| 8      | 128.6 trillion                | 296.2 trillion                | 1.78 trillion                    | 4.1 trillion                     | 29.7 hr          | 24.7 min       |
| 9      | 9 quadrillion                 | 20 quadrillion                | 60 trillion                      | 140 trillion                     | 86.8 days        | 13.9 hr        |

*On avg for 1x RTX 4090.  
**≈63%

#### Note: None of this is guaranteed; it can theoretically take an infinite amount of time to find any vanity address. Have this in mind.

## Contributions

Yes

## Performance

**Update@06/27/25:** Ran a 14x RTX 4090 setup and it did 1.2-1.3 billion searches per second per GPU

## Acknowledgements, External Libraries

- The sha2 implementation used in this library is taken from [here](https://github.com/mochimodev/cuda-hashing-algos), which is in the public domain.
- The base58 encoding implementation is taken from firedancer with heavy modifications for use in cuda & case insensitive encodings, licensed under APACHE-2.0
