# Vanity Server API

A blazingly fast REST API for grinding Solana vanity addresses synchronously.

## Quick Start

```bash
# Build and start server
cargo build --release --features server
cargo run --release --features server -- server

# Configure via .env file
cp env.example .env
# Edit .env with your settings

# Grind addresses
curl -X GET 'http://localhost:8080/grind?base=3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd'
```

## Configuration

All parameters are configured via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `VANITY_PORT` | Server port | `8080` |
| `VANITY_DEFAULT_TOKEN_PROGRAM` | Owner pubkey (required) | - |
| `VANITY_DEFAULT_PREFIX` | Target prefix | - |
| `VANITY_DEFAULT_SUFFIX` | Target suffix | - |
| `VANITY_DEFAULT_CPUS` | CPU threads (0=auto) | `0` |
| `VANITY_DEFAULT_CASE_INSENSITIVE` | Case insensitive | `false` |
| `VANITY_CORS_ORIGINS` | CORS allowed origins (comma-separated) | Permissive (all origins) |

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/health` | GET | Health check |
| `/grind?base=<base>&suffix=<target>[&owner=<program>]` | GET | Grind vanity addresses (synchronous) |

### Grind Vanity Addresses
**GET** `/grind?base=<base>&suffix=<target>`

Returns vanity address result immediately using query parameters and environment variable configuration.

**Query Parameters:**
- `base` (required): Base pubkey for grinding
- `suffix` (optional): Target suffix for vanity addresses
- `owner` (optional): Program that will own the account created with the
  returned seed. Defaults to `VANITY_DEFAULT_TOKEN_PROGRAM`. The owner is part
  of the `create_with_seed` derivation, so a Token-2022 mint must be ground
  with `owner=TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb`; a seed ground for
  the SPL Token program derives a different address under Token-2022.

**Response:**
```json
{
  "address": "H4rHNpqtJUZVRotbSxXTs8oWsL47V7wgPDxJAiuAomni",
  "seed": "gOUdv5rq5lf3Im0r",
  "seed_bytes": [103, 79, 85, 100, 118, 53, 114, 113, 53, 108, 102, 51, 73, 109, 48, 114],
  "base": "3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd",
  "owner": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
  "prefix": null,
  "suffix": "omni",
  "case_insensitive": false,
  "attempts": 918349,
  "duration_seconds": 0.250335924,
  "attempts_per_second": 3668466
}
```

## Examples

### curl
```bash
curl -X GET 'http://localhost:8080/grind?base=3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd&suffix=omni'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8080/grind?base=3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd&suffix=omni');
const result = await response.json();
console.log('Address:', result.address);
console.log('Seed:', result.seed);
console.log('Seed bytes:', result.seed_bytes);
```

### Python
```python
import requests
response = requests.get('http://localhost:8080/grind', params={
    'base': '3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd',
    'suffix': 'omni'
})
result = response.json()
print(f"Address: {result['address']}")
print(f"Seed: {result['seed']}")
print(f"Seed bytes: {result['seed_bytes']}")
```

## Deployment

```bash
# Build
cargo build --release --features server

# Run on VPS
./target/release/vanity server --port 8080

# Use process manager (systemd, PM2, Docker)
# Monitor CPU usage - grinding is intensive
```
