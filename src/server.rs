#[cfg(feature = "server")]
use {
    axum::{
        extract::{Query, State},
        http::StatusCode,
        response::Json,
        routing::get,
        Router,
    },
    serde::{Deserialize, Serialize},
    tower_http::cors::{CorsLayer, AllowOrigin},
};

#[cfg(feature = "server")]
use {
    crate::{array, maybe_bs58_aware_lowercase, parse_pubkey},
    dotenvy,
    fd_bs58,
    rand::{self, Rng},
    rayon,
    sha2::{Digest, Sha256},
    solana_pubkey::Pubkey,
};

#[cfg(feature = "server")]
#[derive(Debug, Deserialize)]
pub struct GrindQuery {
    pub base: String,
    pub suffix: Option<String>,
    /// Program that will own the created account. Defaults to
    /// `VANITY_DEFAULT_TOKEN_PROGRAM`. Token-2022 mints must pass
    /// `TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb`, because the owner is
    /// hashed into the derived address.
    pub owner: Option<String>,
}

#[cfg(feature = "server")]
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GrindResult {
    pub address: String,
    pub seed: String,
    pub seed_bytes: Vec<u8>,
    pub base: String,
    pub owner: String,
    pub prefix: Option<String>,
    pub suffix: Option<String>,
    pub case_insensitive: bool,
    pub attempts: u64,
    pub duration_seconds: f64,
    pub attempts_per_second: u64,
}

#[cfg(feature = "server")]
#[derive(Debug, Clone)]
pub struct AppState;

#[cfg(feature = "server")]
impl AppState {
    pub fn new() -> Self {
        Self
    }
}

#[cfg(feature = "server")]
fn configure_cors() -> CorsLayer {
    let cors_origins = std::env::var("VANITY_CORS_ORIGINS").ok();
    
    if let Some(origins_str) = cors_origins {
        // Parse comma-separated origins
        let origins: Vec<String> = origins_str
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        
        // Parse origins into HeaderValues, filtering out invalid ones
        let origin_list: Vec<axum::http::HeaderValue> = origins
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect();
        
        if origin_list.is_empty() {
            // No valid origins, use permissive CORS
            CorsLayer::permissive()
        } else {
            // Build CORS layer with allowed origins
            let mut cors = CorsLayer::new()
                .allow_methods(tower_http::cors::AllowMethods::any())
                .allow_headers(tower_http::cors::AllowHeaders::any());
            
            // Add origins - use exact for single, list for multiple
            if origin_list.len() == 1 {
                cors = cors.allow_origin(AllowOrigin::exact(origin_list[0].clone()));
            } else {
                cors = cors.allow_origin(AllowOrigin::list(origin_list));
            }
            
            cors
        }
    } else {
        // No CORS config, use permissive CORS
        CorsLayer::permissive()
    }
}

#[cfg(feature = "server")]
pub async fn start_server(args: crate::ServerArgs) -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok(); // Ignore errors if .env file doesn't exist

    // Determine port: CLI arg > env var > default
    let port = args.port
        .or_else(|| std::env::var("VANITY_PORT").ok().and_then(|s| s.parse().ok()))
        .unwrap_or(8080);

    let app_state = AppState::new();

    // Configure CORS
    let cors_layer = configure_cors();

    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/grind", get(grind_sync))
        .layer(cors_layer)
        .with_state(app_state);

    let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;
    println!("🚀 Vanity server running on http://0.0.0.0:{}", port);
    println!("📖 API Documentation:");
    println!("  GET  / - API documentation");
    println!("  GET  /grind - Grind vanity addresses (base required as query param)");
    println!("  GET  /health - Health check");

    axum::serve(listener, app).await?;
    Ok(())
}

#[cfg(feature = "server")]
async fn root() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "service": "vanity-server",
        "version": env!("CARGO_PKG_VERSION"),
        "description": "A blazingly fast tool for grinding vanity addresses on Solana",
        "endpoints": {
            "GET /": "API documentation (this endpoint)",
            "GET /health": "Health check",
            "GET /grind": "Grind vanity addresses synchronously"
        },
        "configuration": {
            "note": "Base is required as a query parameter. Other parameters can be configured via environment variables.",
            "required_vars": [
                "VANITY_DEFAULT_TOKEN_PROGRAM"
            ],
            "optional_vars": [
                "VANITY_DEFAULT_PREFIX",
                "VANITY_DEFAULT_CPUS",
                "VANITY_DEFAULT_CASE_INSENSITIVE",
                "VANITY_PORT",
                "VANITY_CORS_ORIGINS"
            ],
            "query_params": [
                "base - Base pubkey for grinding (required)",
                "suffix - Target suffix for vanity addresses (optional)",
                "owner - Program that will own the created account (optional; defaults to VANITY_DEFAULT_TOKEN_PROGRAM). Token-2022 mints pass TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"
            ]
        },
        "example_usage": {
            "curl": "curl -X GET 'http://localhost:8080/grind?base=3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd&suffix=omni'",
            "curl_token_2022": "curl -X GET 'http://localhost:8080/grind?base=3tJrAXnjofAw8oskbMaSo9oMAYuzdBgVbW3TvQLdMEBd&suffix=yLP&owner=TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb'",
            "description": "Returns vanity address result immediately"
        },
        "response_format": {
            "address": "Generated vanity address",
            "seed": "Seed used to generate the address (string)",
            "seed_bytes": "Seed used to generate the address (byte array)",
            "base": "Base pubkey used",
            "owner": "Owner pubkey used",
            "prefix": "Target prefix (if specified)",
            "suffix": "Target suffix (if specified)",
            "case_insensitive": "Whether case-insensitive matching was used",
            "attempts": "Number of attempts made",
            "duration_seconds": "Time taken in seconds",
            "attempts_per_second": "Performance metric"
        }
    }))
}

#[cfg(feature = "server")]
async fn health_check() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "healthy",
        "service": "vanity-server",
        "version": env!("CARGO_PKG_VERSION")
    }))
}

#[cfg(feature = "server")]
async fn grind_sync(
    Query(query): Query<GrindQuery>,
    State(_state): State<AppState>,
) -> Result<Json<GrindResult>, (StatusCode, Json<serde_json::Value>)> {
    // Get configuration from query parameters and environment variables
    let base_str = query.base;
    let owner_from_query = query.owner.filter(|owner| !owner.trim().is_empty());
    let owner_str = owner_from_query.clone().unwrap_or_else(|| {
        std::env::var("VANITY_DEFAULT_TOKEN_PROGRAM")
            .unwrap_or_else(|_| "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA".to_string())
    });
    let prefix = std::env::var("VANITY_DEFAULT_PREFIX").ok();
    let suffix = query.suffix;
    let case_insensitive = std::env::var("VANITY_DEFAULT_CASE_INSENSITIVE")
        .unwrap_or_else(|_| "false".to_string())
        .parse()
        .unwrap_or(false);
    let num_cpus = std::env::var("VANITY_DEFAULT_CPUS")
        .unwrap_or_else(|_| "0".to_string())
        .parse()
        .unwrap_or(0);

    // Validate pubkeys
    let base = match parse_pubkey(&base_str) {
        Ok(pk) => pk,
        Err(e) => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid base pubkey in query parameter: {}", e)
                })),
            ));
        }
    };

    let owner = match parse_pubkey(&owner_str) {
        Ok(pk) => pk,
        Err(e) if owner_from_query.is_some() => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({
                    "error": format!("Invalid owner pubkey in query parameter: {}", e)
                })),
            ));
        }
        Err(e) => {
            return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({
                    "error": format!("Invalid owner pubkey in config: {}", e)
                })),
            ));
        }
    };

    // Validate prefix/suffix
    let prefix_validated = validate_target(&prefix, case_insensitive);
    let suffix_validated = validate_target(&suffix, case_insensitive);

    // Run grinding synchronously
    let result = tokio::task::spawn_blocking(move || {
        grind_sync_blocking(
            base,
            owner,
            prefix_validated,
            suffix_validated,
            case_insensitive,
            num_cpus,
        )
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Grinding task failed: {}", e)
            })),
        )
    })?;

    match result {
        Ok(grind_result) => Ok(Json(grind_result)),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "error": format!("Grinding failed: {}", e)
            })),
        )),
    }
}

#[cfg(feature = "server")]
fn grind_sync_blocking(
    base: Pubkey,
    owner: Pubkey,
    prefix: Option<String>,
    suffix: Option<String>,
    case_insensitive: bool,
    num_cpus: u32,
) -> Result<GrindResult, String> {
    let start_time = std::time::Instant::now();
    let mut num_cpus = num_cpus;
    
    // Auto-detect CPU threads if 0
    if num_cpus == 0 {
        num_cpus = rayon::current_num_threads() as u32;
    }

    let prefix_str = prefix.as_deref().unwrap_or("").to_string();
    let suffix_str = suffix.as_deref().unwrap_or("").to_string();
    let exit_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let result_arc = std::sync::Arc::new(std::sync::Mutex::new(None));

    // Start CPU grinding
    let base_clone = base;
    let owner_clone = owner;
    let prefix_clone = prefix.clone();
    let suffix_clone = suffix.clone();
    let exit_flag_clone = exit_flag.clone();
    let result_clone = result_arc.clone();

    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus as usize)
        .build_global()
        .unwrap_or_else(|_| {
            // Thread pool already initialized, continue with current settings
        });

    let cpu_handle = std::thread::spawn(move || {
        let base_sha = Sha256::new().chain_update(base_clone);
        let mut count = 0_u64;

        loop {
            if exit_flag_clone.load(std::sync::atomic::Ordering::Acquire) {
                return;
            }

            let mut seed_iter = rand::thread_rng().sample_iter(&rand::distributions::Alphanumeric).take(16);
            let seed: [u8; 16] = array::from_fn(|_| seed_iter.next().unwrap());

            let pubkey_bytes: [u8; 32] = base_sha
                .clone()
                .chain_update(seed)
                .chain_update(owner_clone)
                .finalize()
                .into();
            let pubkey = fd_bs58::encode_32(pubkey_bytes);
            let out_str_target_check = maybe_bs58_aware_lowercase(&pubkey, case_insensitive);

            count += 1;

            // Check if we found the target
            if out_str_target_check.starts_with(&prefix_str) && out_str_target_check.ends_with(&suffix_str) {
                let time_secs = start_time.elapsed().as_secs_f64();
                let attempts_per_second = (count as f64 / time_secs) as u64;

                let result = GrindResult {
                    address: pubkey,
                    seed: String::from_utf8_lossy(&seed).to_string(),
                    seed_bytes: seed.to_vec(),
                    base: base_clone.to_string(),
                    owner: owner_clone.to_string(),
                    prefix: prefix_clone,
                    suffix: suffix_clone,
                    case_insensitive,
                    attempts: count,
                    duration_seconds: time_secs,
                    attempts_per_second,
                };

                *result_clone.lock().unwrap() = Some(result);
                exit_flag_clone.store(true, std::sync::atomic::Ordering::Release);
                return;
            }
        }
    });

    // Wait for completion or timeout
    let result = cpu_handle.join();

    match result {
        Ok(_) => {
            if let Some(grind_result) = result_arc.lock().unwrap().take() {
                Ok(grind_result)
            } else {
                Err("Grinding timed out after 5 minutes".to_string())
            }
        }
        Err(_) => Err("Grinding thread panicked".to_string()),
    }
}


#[cfg(feature = "server")]
fn validate_target(target: &Option<String>, case_insensitive: bool) -> Option<String> {
    const BS58_CHARS: &str = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    if let Some(ref target_str) = target {
        for c in target_str.chars() {
            if !BS58_CHARS.contains(c) {
                return None;
            }
        }
        Some(maybe_bs58_aware_lowercase(target_str, case_insensitive))
    } else {
        None
    }
}
