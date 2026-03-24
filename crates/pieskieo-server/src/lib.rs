use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::net::{IpAddr, SocketAddr};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use axum::extract::{
    ws::{Message, WebSocket, WebSocketUpgrade},
    DefaultBodyLimit,
};
use axum::{
    body::Body,
    extract::{ConnectInfo, Path, Query, State},
    http::Request,
    middleware::{self, Next},
    response::IntoResponse,
    routing::{delete, get, post},
    Extension, Json, Router,
};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use futures::future::join_all;
use pieskieo_core::{
    PieskieoDb, PieskieoError, SchemaDef, SchemaField, SqlResult,
    VectorParams as PieskieoVectorParams,
};
use rand::{distributions::Alphanumeric, Rng};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use sqlparser::{dialect::GenericDialect, parser::Parser};
use std::path::PathBuf;
use tokio::{net::TcpListener, sync::RwLock};
use tracing_appender::rolling;
use tracing_subscriber::fmt::writer::MakeWriterExt;
use tracing_subscriber::{prelude::*, EnvFilter};
use uuid::Uuid;

#[cfg(feature = "tls")]
use {
    axum_server::tls_rustls::RustlsConfig,
    rustls::{Certificate, PrivateKey},
    rustls_pemfile::{certs, read_all, Item},
    std::fs::File,
    std::io::BufReader,
};

#[derive(Clone)]
struct AppState {
    pool: Arc<RwLock<DbPool>>,
    auth: Arc<RwLock<AuthConfig>>,
    limiter: Arc<RateLimiter>,
    audit: Arc<AuditLog>,
    data_dir: String,
    pause_writes: Arc<AtomicBool>,
    reshard_status: Arc<RwLock<Option<ReshardReport>>>,
}

#[derive(Clone, Serialize, Deserialize)]
struct ReshardReport {
    status: String,
    paused: bool,
    verified: bool,
    before_counts: HashMap<usize, usize>,
    after_counts: HashMap<usize, usize>,
}

#[derive(Clone)]
struct AuthConfig {
    users: Vec<UserRec>,
    bearer: Option<String>,
    path: PathBuf,
    attempts: Arc<Mutex<HashMap<String, Attempt>>>,
    max_failures: u32,
    lockout: Duration,
    window: Duration,
}

struct RateLimiter {
    window: Duration,
    max: u32,
    hits: Arc<Mutex<HashMap<IpAddr, (u32, Instant)>>>,
    rejected: AtomicU64,
}

#[derive(Clone)]
struct AuditLog {
    path: PathBuf,
}

impl AuditLog {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn write(
        &self,
        ip: IpAddr,
        method: &str,
        path: &str,
        status: u16,
        role: Option<Role>,
        dur: Duration,
    ) {
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let role = role
            .map(|r| match r {
                Role::Admin => "admin",
                Role::Write => "write",
                Role::Read => "read",
            })
            .unwrap_or("anon")
            .to_string();
        let line = format!(
            "{ts},{ip},{method},{path},{status},{role},{:.3}\n",
            dur.as_secs_f64() * 1000.0
        );
        let target = self.rotate_target();
        tokio::task::spawn_blocking(move || {
            let _ = ensure_parent(&target);
            if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&target) {
                let _ = f.write_all(line.as_bytes());
            }
        });
    }

    fn rotate_target(&self) -> PathBuf {
        // daily files and size-based rollover
        let max_mb = std::env::var("PIESKIEO_AUDIT_MAX_MB")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10);
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let base = self
            .path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_else(|| "audit.log".into());
        let dir = self
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| PathBuf::from("."));
        let candidate = dir.join(format!("{today}-{base}"));
        let limit = max_mb * 1024 * 1024;
        if let Ok(meta) = std::fs::metadata(&candidate) {
            if meta.len() >= limit {
                let ts = chrono::Utc::now().format("%H%M%S").to_string();
                return dir.join(format!("{today}-{ts}-{base}"));
            }
        }
        candidate
    }
}

fn ensure_parent(path: &PathBuf) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}
#[derive(Clone)]
struct UserRec {
    user: String,
    password_hash: String,
    role: Role,
}

#[derive(Clone, Debug)]
struct Attempt {
    count: u32,
    first: Instant,
    locked_until: Option<Instant>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Role {
    Admin,
    Write,
    Read,
}

impl AuthConfig {
    fn from_env(data_dir: &str) -> Self {
        // primary multi-user source: PIESKIEO_USERS as JSON array [{user,pass,role}]
        let mut users = Vec::new();
        let path = PathBuf::from(data_dir).join("auth_users.json");
        let max_failures = std::env::var("PIESKIEO_AUTH_MAX_FAILURES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);
        let lockout = std::env::var("PIESKIEO_AUTH_LOCKOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(300));
        let window = std::env::var("PIESKIEO_AUTH_WINDOW_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(900));
        if let Ok(json) = std::env::var("PIESKIEO_USERS") {
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&json) {
                if let Some(arr) = val.as_array() {
                    for item in arr {
                        if let (Some(u), Some(p)) = (
                            item.get("user").and_then(|v| v.as_str()),
                            item.get("pass").and_then(|v| v.as_str()),
                        ) {
                            if let Err(msg) = Self::validate_password(p) {
                                tracing::warn!("skipping user {} from PIESKIEO_USERS: {}", u, msg);
                                continue;
                            }
                            let role = item
                                .get("role")
                                .and_then(|v| v.as_str())
                                .map(Self::parse_role)
                                .unwrap_or(Role::Read);
                            users.push(UserRec {
                                user: u.to_string(),
                                password_hash: Self::hash_password(p),
                                role,
                            });
                        }
                    }
                }
            }
        }
        // fallback single user env
        if users.is_empty() {
            if let Ok(u) = std::env::var("PIESKIEO_AUTH_USER") {
                if let Ok(p) = std::env::var("PIESKIEO_AUTH_PASSWORD") {
                    if let Err(msg) = Self::validate_password(&p) {
                        tracing::warn!("skipping PIESKIEO_AUTH_USER {}: {}", u, msg);
                    } else {
                        users.push(UserRec {
                            user: u,
                            password_hash: Self::hash_password(&p),
                            role: Role::Admin,
                        });
                    }
                }
            }
        }
        // file-based persistent users
        if users.is_empty() && path.exists() {
            if let Ok(txt) = std::fs::read_to_string(&path) {
                // Only parse if file has content
                if !txt.trim().is_empty() {
                    if let Ok(arr) = serde_json::from_str::<Vec<UserDisk>>(&txt) {
                        for u in arr {
                            users.push(UserRec {
                                user: u.user,
                                password_hash: u.pass,
                                role: Self::parse_role(&u.role),
                            });
                        }
                    } else {
                        tracing::warn!("auth_users.json exists but contains invalid JSON; will use default admin");
                    }
                } else {
                    tracing::warn!("auth_users.json exists but is empty; will use default admin");
                }
            }
        }
        let bearer = std::env::var("PIESKIEO_TOKEN").ok();
        let mut default_admin_generated = false;
        // default admin if nothing configured (even if auth_users.json exists but was empty/invalid)
        if users.is_empty() && bearer.is_none() {
            let pass = Self::generate_secure_password();
            tracing::warn!("No users configured; generated secure default admin password: {pass}");
            tracing::warn!("THIS PASSWORD HAS BEEN PERSISTED to auth_users.json. PLEASE CHANGE IT IMMEDIATELY.");
            users.push(UserRec {
                user: "admin".into(),
                password_hash: Self::hash_password(&pass),
                role: Role::Admin,
            });
            default_admin_generated = true;
        }
        let config = Self {
            users,
            bearer,
            path,
            attempts: Arc::new(Mutex::new(HashMap::new())),
            max_failures,
            lockout,
            window,
        };
        if default_admin_generated {
            config.persist();
        }
        config
    }

    fn enabled(&self) -> bool {
        !self.users.is_empty() || self.bearer.is_some()
    }

    fn parse_role(s: &str) -> Role {
        match s.to_ascii_lowercase().as_str() {
            "admin" => Role::Admin,
            "write" | "writer" => Role::Write,
            _ => Role::Read,
        }
    }

    fn validate_password(pass: &str) -> Result<(), String> {
        if pass.len() < 8 {
            return Err("password must be at least 8 characters".into());
        }
        let mut has_upper = false;
        let mut has_lower = false;
        let mut has_digit = false;
        let mut has_symbol = false;
        for ch in pass.chars() {
            if ch.is_ascii_uppercase() {
                has_upper = true;
            } else if ch.is_ascii_lowercase() {
                has_lower = true;
            } else if ch.is_ascii_digit() {
                has_digit = true;
            } else {
                has_symbol = true;
            }
        }
        if !(has_upper && has_lower && has_digit && has_symbol) {
            return Err("password must contain upper, lower, digit, and symbol characters".into());
        }
        Ok(())
    }

    fn persist(&self) {
        let disk: Vec<UserDisk> = self
            .users
            .iter()
            .map(|u| UserDisk {
                user: u.user.clone(),
                pass: u.password_hash.clone(),
                role: match u.role {
                    Role::Admin => "admin",
                    Role::Write => "write",
                    Role::Read => "read",
                }
                .to_string(),
            })
            .collect();
        if let Ok(txt) = serde_json::to_string_pretty(&disk) {
            let _ = std::fs::create_dir_all(
                self.path
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new(".")),
            );
            let _ = std::fs::write(&self.path, txt);
        }
    }

    fn generate_secure_password() -> String {
        let mut rng = rand::thread_rng();
        let mut pass: String = rng
            .sample_iter(&Alphanumeric)
            .take(28)
            .map(char::from)
            .collect();

        // Ensure complexity requirements: upper, lower, digit, symbol
        pass.push(rng.gen_range('A'..='Z'));
        pass.push(rng.gen_range('a'..='z'));
        pass.push(rng.gen_range('0'..='9'));
        pass.push('!'); // simplest symbol

        pass
    }

    fn hash_password(pass: &str) -> String {
        let salt = argon2::password_hash::SaltString::generate(&mut OsRng);
        let argon = Argon2::new(
            argon2::Algorithm::Argon2id,
            argon2::Version::V0x13,
            argon2::Params::new(19 * 1024, 3, 1, None).unwrap(), // ~19 MiB, 3 iterations
        );
        argon
            .hash_password(pass.as_bytes(), &salt)
            .map(|h| h.to_string())
            .unwrap_or_default()
    }

    fn verify_password(hash: &str, pass: &str) -> bool {
        if let Ok(parsed) = PasswordHash::new(hash) {
            Argon2::default()
                .verify_password(pass.as_bytes(), &parsed)
                .is_ok()
        } else {
            false
        }
    }

    fn check_lockout(&self, user: &str) -> bool {
        let mut map = self.attempts.lock().unwrap();
        if let Some(state) = map.get_mut(user) {
            let now = Instant::now();
            if let Some(until) = state.locked_until {
                if now < until {
                    return true;
                } else {
                    state.locked_until = None;
                    state.count = 0;
                    state.first = now;
                }
            }
            if now.duration_since(state.first) > self.window {
                state.count = 0;
                state.first = now;
            }
        }
        false
    }

    fn record_failure(&self, user: &str) {
        let mut map = self.attempts.lock().unwrap();
        let now = Instant::now();
        let entry = map.entry(user.to_string()).or_insert(Attempt {
            count: 0,
            first: now,
            locked_until: None,
        });
        if now.duration_since(entry.first) > self.window {
            entry.count = 0;
            entry.first = now;
        }
        entry.count += 1;
        if entry.count >= self.max_failures {
            entry.locked_until = Some(now + self.lockout);
            tracing::warn!(
                "user {} locked out for {:?} after {} failures",
                user,
                self.lockout,
                entry.count
            );
        }
    }

    fn record_success(&self, user: &str) {
        let mut map = self.attempts.lock().unwrap();
        map.remove(user);
    }
}

impl RateLimiter {
    fn from_env() -> Self {
        let max = std::env::var("PIESKIEO_RATE_MAX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(300);
        let window = std::env::var("PIESKIEO_RATE_WINDOW_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(60));
        Self {
            window,
            max,
            hits: Arc::new(Mutex::new(HashMap::new())),
            rejected: AtomicU64::new(0),
        }
    }

    fn allow(&self, ip: IpAddr) -> Result<(), Duration> {
        let now = Instant::now();
        let mut hits = self.hits.lock().unwrap();
        let entry = hits.entry(ip).or_insert((0, now));
        let (ref mut count, ref mut start) = *entry;
        if now.duration_since(*start) > self.window {
            *count = 0;
            *start = now;
        }
        *count += 1;
        if *count <= self.max {
            Ok(())
        } else {
            self.rejected.fetch_add(1, Ordering::Relaxed);
            Err(self.window.saturating_sub(now.duration_since(*start)))
        }
    }
}

#[derive(Serialize, Deserialize)]
struct UserDisk {
    user: String,
    pass: String,
    role: String,
}

#[derive(Deserialize)]
struct DocInput {
    id: Option<Uuid>,
    data: serde_json::Value,
    namespace: Option<String>,
    collection: Option<String>,
}

#[derive(Deserialize)]
struct QueryInput {
    filter: HashMap<String, serde_json::Value>,
    limit: Option<usize>,
    namespace: Option<String>,
    collection: Option<String>,
    table: Option<String>,
    offset: Option<usize>,
    sql: Option<String>,
}

#[derive(Deserialize)]
struct SqlInput {
    sql: String,
    limit: Option<usize>,
}

#[derive(Deserialize)]
struct SchemaInput {
    family: String, // "doc" or "row"
    namespace: Option<String>,
    name: String,
    fields: HashMap<String, SchemaField>,
}

#[derive(Deserialize)]
struct UserCreateInput {
    user: String,
    pass: String,
    role: Option<String>,
}

#[derive(Deserialize)]
struct VectorInput {
    id: Uuid,
    vector: Vec<f32>,
    meta: Option<HashMap<String, String>>,
    namespace: Option<String>,
}

#[derive(Deserialize)]
struct VectorBulk {
    items: Vec<VectorInput>,
}

#[derive(Deserialize)]
struct VectorMetaInput {
    meta: HashMap<String, String>,
}

#[derive(Deserialize)]
struct VectorMetaDeleteInput {
    keys: Vec<String>,
}

#[derive(Deserialize)]
struct RowInput {
    id: Option<Uuid>,
    data: serde_json::Value,
    namespace: Option<String>,
    table: Option<String>,
}

#[derive(Deserialize)]
struct VectorSearchInput {
    query: Vec<f32>,
    k: Option<usize>,
    metric: Option<String>,
    filter_ids: Option<Vec<Uuid>>,
    ef_search: Option<usize>,
    filter_meta: Option<HashMap<String, String>>,
    namespace: Option<String>,
}

#[derive(Deserialize)]
struct VectorConfigInput {
    ef_search: Option<usize>,
    ef_construction: Option<usize>,
    link_top_k: Option<usize>,
}

#[derive(Deserialize)]
struct ReplicationBatch {
    records: Vec<String>, // base64-encoded RecordKind
}

#[derive(Deserialize)]
struct ReshardRequest {
    shards: usize,
}

#[derive(Serialize)]
struct ReshardStatus {
    status: Option<ReshardReport>,
    paused: bool,
}
#[derive(Deserialize)]
struct WalQuery {
    since: Option<u64>,
}

#[derive(Serialize)]
struct WalShardSlice {
    shard: usize,
    end_offset: u64,
    records: Vec<String>,
}

#[derive(Serialize)]
struct WalExport {
    slices: Vec<WalShardSlice>,
}

#[derive(Serialize)]
struct WalStream {
    end_offset: u64,
    slices: Vec<WalShardSlice>,
}
#[derive(Deserialize)]
struct EdgeInput {
    src: Uuid,
    dst: Uuid,
    weight: Option<f32>,
}

#[derive(Serialize)]
struct ApiResponse<T> {
    ok: bool,
    data: T,
}

#[derive(Deserialize)]
struct NsParams {
    namespace: Option<String>,
    collection: Option<String>,
    table: Option<String>,
}

#[derive(Serialize)]
struct VectorOutput {
    id: Uuid,
    vector: Vec<f32>,
    meta: Option<HashMap<String, String>>,
}

fn extract_insert_id(stmt: &sqlparser::ast::Statement) -> Result<Option<Uuid>, ApiError> {
    let sqlparser::ast::Statement::Insert {
        columns, source, ..
    } = stmt
    else {
        return Ok(None);
    };

    let columns: Vec<String> = columns.iter().map(|c| c.value.clone()).collect();
    let query = source
        .as_ref()
        .ok_or_else(|| ApiError::BadRequest("INSERT source required".into()))?;
    let values = match query.body.as_ref() {
        sqlparser::ast::SetExpr::Values(v) => &v.rows,
        _ => {
            return Err(ApiError::BadRequest(
                "only VALUES inserts are supported in /v1/sql".into(),
            ))
        }
    };

    if values.len() != 1 {
        return Err(ApiError::BadRequest(
            "multi-row INSERT is not supported in /v1/sql".into(),
        ));
    }

    for (idx, column) in columns.iter().enumerate() {
        if column != "id" && column != "_id" {
            continue;
        }
        let Some(expr) = values[0].get(idx) else {
            continue;
        };
        let sqlparser::ast::Expr::Value(sqlparser::ast::Value::SingleQuotedString(raw)) = expr
        else {
            return Err(ApiError::BadRequest(
                "sharded /v1/sql INSERT requires id/_id to be a quoted UUID literal".into(),
            ));
        };
        let id = Uuid::parse_str(raw)
            .map_err(|_| ApiError::BadRequest("INSERT id/_id must be a valid UUID".into()))?;
        return Ok(Some(id));
    }

    Ok(None)
}

fn strip_select_limit_offset(stmt: &sqlparser::ast::Statement) -> Result<Option<String>, ApiError> {
    let sqlparser::ast::Statement::Query(query) = stmt else {
        return Ok(None);
    };
    let mut stmt = stmt.clone();
    let sqlparser::ast::Statement::Query(query) = &mut stmt else {
        unreachable!();
    };
    query.limit = None;
    query.offset = None;
    Ok(Some(stmt.to_string()))
}

fn json_cmp(a: &serde_json::Value, b: &serde_json::Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (serde_json::Value::Number(x), serde_json::Value::Number(y)) => {
            let xf = x.as_f64()?;
            let yf = y.as_f64()?;
            xf.partial_cmp(&yf)
        }
        (serde_json::Value::String(x), serde_json::Value::String(y)) => Some(x.cmp(y)),
        (serde_json::Value::Bool(x), serde_json::Value::Bool(y)) => Some(x.cmp(y)),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum AggregateKind {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

#[derive(Clone)]
struct GroupField {
    source: String,
    alias: String,
}

#[derive(Clone)]
struct AggregateSpec {
    alias: String,
    kind: AggregateKind,
    field: Option<String>,
}

#[derive(Clone)]
struct AggregatePlan {
    group_fields: Vec<GroupField>,
    specs: Vec<AggregateSpec>,
    having: Option<sqlparser::ast::Expr>,
}

fn parse_simple_aggregate_plan(
    stmt: &sqlparser::ast::Statement,
) -> Result<Option<AggregatePlan>, ApiError> {
    let sqlparser::ast::Statement::Query(query) = stmt else {
        return Ok(None);
    };
    let sqlparser::ast::SetExpr::Select(select) = query.body.as_ref() else {
        return Ok(None);
    };

    let group_sources = match &select.group_by {
        sqlparser::ast::GroupByExpr::All(_) => {
            return Err(ApiError::BadRequest(
                "GROUP BY ALL is not supported on /v1/sql".into(),
            ))
        }
        sqlparser::ast::GroupByExpr::Expressions(exprs) => {
            let mut out = std::collections::HashSet::new();
            for expr in exprs {
                let sqlparser::ast::Expr::Identifier(id) = expr else {
                    return Err(ApiError::BadRequest(
                        "cross-shard GROUP BY supports only identifiers".into(),
                    ));
                };
                out.insert(id.value.clone());
            }
            out
        }
    };

    let mut group_fields = Vec::new();
    let mut specs = Vec::new();
    for item in &select.projection {
        match item {
            sqlparser::ast::SelectItem::UnnamedExpr(sqlparser::ast::Expr::Identifier(id)) => {
                if !group_sources.contains(&id.value) {
                    return Ok(None);
                }
                group_fields.push(GroupField {
                    source: id.value.clone(),
                    alias: id.value.clone(),
                });
            }
            sqlparser::ast::SelectItem::ExprWithAlias {
                expr: sqlparser::ast::Expr::Identifier(id),
                alias,
            } => {
                if !group_sources.contains(&id.value) {
                    return Ok(None);
                }
                group_fields.push(GroupField {
                    source: id.value.clone(),
                    alias: alias.value.clone(),
                });
            }
            sqlparser::ast::SelectItem::UnnamedExpr(sqlparser::ast::Expr::Function(f)) => {
                specs.push(parse_aggregate_spec(f, None)?);
            }
            sqlparser::ast::SelectItem::ExprWithAlias {
                expr: sqlparser::ast::Expr::Function(f),
                alias,
            } => {
                specs.push(parse_aggregate_spec(f, Some(alias.value.as_str()))?);
            }
            _ => return Ok(None),
        }
    }
    if specs.is_empty() {
        return Ok(None);
    }
    Ok(Some(AggregatePlan {
        group_fields,
        specs,
        having: select.having.clone(),
    }))
}

fn parse_aggregate_spec(
    func: &sqlparser::ast::Function,
    alias: Option<&str>,
) -> Result<AggregateSpec, ApiError> {
    let name = func.name.to_string().to_lowercase();
    let kind = match name.as_str() {
        "count" => AggregateKind::Count,
        "sum" => AggregateKind::Sum,
        "avg" => AggregateKind::Avg,
        "min" => AggregateKind::Min,
        "max" => AggregateKind::Max,
        _ => {
            return Err(ApiError::BadRequest(format!(
                "cross-shard aggregate {} is not yet supported on /v1/sql",
                name
            )))
        }
    };

    let mut field = None;
    if let Some(arg) = func.args.first() {
        match arg {
            sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Wildcard) => {
                if !matches!(kind, AggregateKind::Count) {
                    return Err(ApiError::BadRequest(
                        "only COUNT supports wildcard arguments on /v1/sql".into(),
                    ));
                }
            }
            sqlparser::ast::FunctionArg::Unnamed(sqlparser::ast::FunctionArgExpr::Expr(
                sqlparser::ast::Expr::Identifier(id),
            )) => {
                field = Some(id.value.clone());
            }
            _ => {
                return Err(ApiError::BadRequest(
                    "cross-shard aggregate arguments must be identifiers or *".into(),
                ))
            }
        }
    }

    if !matches!(kind, AggregateKind::Count) && field.is_none() {
        return Err(ApiError::BadRequest(
            "SUM/AVG/MIN/MAX require an identifier argument on /v1/sql".into(),
        ));
    }

    Ok(AggregateSpec {
        alias: alias.unwrap_or(&name).to_string(),
        kind,
        field,
    })
}

fn build_raw_aggregate_fanout_sql(
    stmt: &sqlparser::ast::Statement,
    plan: &AggregatePlan,
) -> Result<Option<String>, ApiError> {
    let sqlparser::ast::Statement::Query(_) = stmt else {
        return Ok(None);
    };
    let mut stmt = stmt.clone();
    let sqlparser::ast::Statement::Query(query) = &mut stmt else {
        unreachable!();
    };
    let sqlparser::ast::SetExpr::Select(select) = query.body.as_mut() else {
        return Err(ApiError::BadRequest(
            "only simple SELECT queries can be merged across shards".into(),
        ));
    };

    let mut seen = std::collections::HashSet::new();
    let mut projection = Vec::new();
    for field in &plan.group_fields {
        if !seen.insert(field.source.clone()) {
            continue;
        }
        projection.push(sqlparser::ast::SelectItem::ExprWithAlias {
            expr: sqlparser::ast::Expr::Identifier(sqlparser::ast::Ident::new(
                field.source.clone(),
            )),
            alias: sqlparser::ast::Ident::new(field.source.clone()),
        });
    }
    for spec in &plan.specs {
        let source = spec.field.clone().unwrap_or_else(|| "_id".to_string());
        if !seen.insert(source.clone()) {
            continue;
        }
        projection.push(sqlparser::ast::SelectItem::ExprWithAlias {
            expr: sqlparser::ast::Expr::Identifier(sqlparser::ast::Ident::new(source.clone())),
            alias: sqlparser::ast::Ident::new(source),
        });
    }
    if projection.is_empty() {
        projection.push(sqlparser::ast::SelectItem::ExprWithAlias {
            expr: sqlparser::ast::Expr::Identifier(sqlparser::ast::Ident::new("_id")),
            alias: sqlparser::ast::Ident::new("_id"),
        });
    }
    select.projection = projection;
    query.order_by.clear();
    query.limit = None;
    query.offset = None;
    Ok(Some(stmt.to_string()))
}

fn compute_simple_aggregate_rows(
    rows: Vec<(Uuid, serde_json::Value)>,
    specs: &[AggregateSpec],
) -> Result<Vec<(Uuid, serde_json::Value)>, ApiError> {
    let mut merged = serde_json::Map::new();
    for spec in specs {
        match spec.kind {
            AggregateKind::Count => {
                let count = if let Some(field) = &spec.field {
                    rows.iter()
                        .filter(|(_, row)| row.get(field).is_some_and(|value| !value.is_null()))
                        .count() as u64
                } else {
                    rows.len() as u64
                };
                merged.insert(spec.alias.clone(), serde_json::json!(count));
            }
            AggregateKind::Sum => {
                let field = spec.field.as_ref().expect("sum field");
                let mut total = 0.0_f64;
                let mut saw = false;
                for (_, row) in &rows {
                    let Some(value) = row.get(field) else {
                        continue;
                    };
                    if value.is_null() {
                        continue;
                    }
                    let num = value.as_f64().ok_or_else(|| {
                        ApiError::Internal(anyhow::anyhow!("invalid SUM input for field {}", field))
                    })?;
                    total += num;
                    saw = true;
                }
                merged.insert(
                    spec.alias.clone(),
                    if saw {
                        serde_json::json!(total)
                    } else {
                        serde_json::Value::Null
                    },
                );
            }
            AggregateKind::Avg => {
                let field = spec.field.as_ref().expect("avg field");
                let mut total = 0.0_f64;
                let mut count = 0usize;
                for (_, row) in &rows {
                    let Some(value) = row.get(field) else {
                        continue;
                    };
                    if value.is_null() {
                        continue;
                    }
                    let num = value.as_f64().ok_or_else(|| {
                        ApiError::Internal(anyhow::anyhow!("invalid AVG input for field {}", field))
                    })?;
                    total += num;
                    count += 1;
                }
                merged.insert(
                    spec.alias.clone(),
                    if count == 0 {
                        serde_json::Value::Null
                    } else {
                        serde_json::json!(total / count as f64)
                    },
                );
            }
            AggregateKind::Min | AggregateKind::Max => {
                let field = spec.field.as_ref().expect("min/max field");
                let mut best: Option<serde_json::Value> = None;
                for (_, row) in &rows {
                    let Some(value) = row.get(field) else {
                        continue;
                    };
                    if value.is_null() {
                        continue;
                    }
                    best = match best {
                        None => Some(value.clone()),
                        Some(current) => {
                            let ord = json_cmp(&current, value).ok_or_else(|| {
                                ApiError::Internal(anyhow::anyhow!(
                                    "invalid aggregate ordering for field {}",
                                    field
                                ))
                            })?;
                            let take_new = matches!(spec.kind, AggregateKind::Min) && ord.is_gt()
                                || matches!(spec.kind, AggregateKind::Max) && ord.is_lt();
                            if take_new {
                                Some(value.clone())
                            } else {
                                Some(current)
                            }
                        }
                    };
                }
                merged.insert(spec.alias.clone(), best.unwrap_or(serde_json::Value::Null));
            }
        }
    }
    Ok(vec![(Uuid::nil(), serde_json::Value::Object(merged))])
}

fn sql_value_to_json(value: &sqlparser::ast::Value) -> Option<serde_json::Value> {
    match value {
        sqlparser::ast::Value::Number(n, _) => {
            if let Ok(i) = n.parse::<i64>() {
                Some(serde_json::json!(i))
            } else if let Ok(f) = n.parse::<f64>() {
                Some(serde_json::json!(f))
            } else {
                None
            }
        }
        sqlparser::ast::Value::SingleQuotedString(s) => Some(serde_json::Value::String(s.clone())),
        sqlparser::ast::Value::Boolean(b) => Some(serde_json::Value::Bool(*b)),
        sqlparser::ast::Value::Null => Some(serde_json::Value::Null),
        _ => None,
    }
}

fn eval_having_operand(
    row: &serde_json::Value,
    expr: &sqlparser::ast::Expr,
) -> Result<serde_json::Value, ApiError> {
    match expr {
        sqlparser::ast::Expr::Identifier(id) => Ok(row
            .get(&id.value)
            .cloned()
            .unwrap_or(serde_json::Value::Null)),
        sqlparser::ast::Expr::Value(v) => sql_value_to_json(v).ok_or_else(|| {
            ApiError::BadRequest("HAVING literal is not supported on /v1/sql".into())
        }),
        sqlparser::ast::Expr::Nested(inner) => eval_having_operand(row, inner),
        _ => Err(ApiError::BadRequest(
            "HAVING supports only identifiers and literals on /v1/sql".into(),
        )),
    }
}

fn eval_having_expr(
    row: &serde_json::Value,
    expr: &sqlparser::ast::Expr,
) -> Result<bool, ApiError> {
    match expr {
        sqlparser::ast::Expr::BinaryOp { left, op, right } => match op {
            sqlparser::ast::BinaryOperator::And => {
                Ok(eval_having_expr(row, left)? && eval_having_expr(row, right)?)
            }
            sqlparser::ast::BinaryOperator::Or => {
                Ok(eval_having_expr(row, left)? || eval_having_expr(row, right)?)
            }
            sqlparser::ast::BinaryOperator::Eq
            | sqlparser::ast::BinaryOperator::NotEq
            | sqlparser::ast::BinaryOperator::Gt
            | sqlparser::ast::BinaryOperator::Lt
            | sqlparser::ast::BinaryOperator::GtEq
            | sqlparser::ast::BinaryOperator::LtEq => {
                let left = eval_having_operand(row, left)?;
                let right = eval_having_operand(row, right)?;
                let ord = json_cmp(&left, &right);
                Ok(match op {
                    sqlparser::ast::BinaryOperator::Eq => left == right,
                    sqlparser::ast::BinaryOperator::NotEq => left != right,
                    sqlparser::ast::BinaryOperator::Gt => ord.map(|o| o.is_gt()).unwrap_or(false),
                    sqlparser::ast::BinaryOperator::Lt => ord.map(|o| o.is_lt()).unwrap_or(false),
                    sqlparser::ast::BinaryOperator::GtEq => ord.map(|o| o.is_ge()).unwrap_or(false),
                    sqlparser::ast::BinaryOperator::LtEq => ord.map(|o| o.is_le()).unwrap_or(false),
                    _ => unreachable!(),
                })
            }
            _ => Err(ApiError::BadRequest(
                "HAVING operator is not supported on /v1/sql".into(),
            )),
        },
        sqlparser::ast::Expr::Nested(inner) => eval_having_expr(row, inner),
        _ => Err(ApiError::BadRequest(
            "HAVING must be a boolean expression on /v1/sql".into(),
        )),
    }
}

fn compute_grouped_aggregate_rows(
    rows: Vec<(Uuid, serde_json::Value)>,
    plan: &AggregatePlan,
) -> Result<Vec<(Uuid, serde_json::Value)>, ApiError> {
    if plan.group_fields.is_empty() {
        let out = compute_simple_aggregate_rows(rows, &plan.specs)?;
        if let Some(expr) = &plan.having {
            let mut filtered = Vec::new();
            for row in out {
                if eval_having_expr(&row.1, expr)? {
                    filtered.push(row);
                }
            }
            return Ok(filtered);
        }
        return Ok(out);
    }

    let mut groups: std::collections::HashMap<
        String,
        (
            serde_json::Map<String, serde_json::Value>,
            Vec<(Uuid, serde_json::Value)>,
        ),
    > = std::collections::HashMap::new();

    for row in rows {
        let mut group_values = serde_json::Map::new();
        if let Some(obj) = row.1.as_object() {
            for field in &plan.group_fields {
                group_values.insert(
                    field.alias.clone(),
                    obj.get(&field.source)
                        .cloned()
                        .unwrap_or(serde_json::Value::Null),
                );
            }
        }
        let key = serde_json::to_string(&group_values)
            .map_err(|e| ApiError::Internal(anyhow::anyhow!(e)))?;
        groups
            .entry(key)
            .or_insert_with(|| (group_values, Vec::new()))
            .1
            .push(row);
    }

    let mut out = Vec::with_capacity(groups.len());
    for (_key, (group_values, bucket_rows)) in groups {
        let mut agg_row = compute_simple_aggregate_rows(bucket_rows, &plan.specs)?
            .into_iter()
            .next()
            .map(|(_, value)| value)
            .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
        if let Some(obj) = agg_row.as_object_mut() {
            for (key, value) in group_values {
                obj.insert(key, value);
            }
        }
        let keep = match &plan.having {
            Some(expr) => eval_having_expr(&agg_row, expr)?,
            None => true,
        };
        if keep {
            out.push((Uuid::nil(), agg_row));
        }
    }
    Ok(out)
}

fn sort_select_rows(
    rows: &mut [(Uuid, serde_json::Value)],
    query: &sqlparser::ast::Query,
) -> Result<(), ApiError> {
    let mut order_fields = Vec::new();
    for ob in &query.order_by {
        let sqlparser::ast::Expr::Identifier(id) = &ob.expr else {
            return Err(ApiError::BadRequest(
                "cross-shard ORDER BY supports only identifiers".into(),
            ));
        };
        order_fields.push((id.value.clone(), ob.asc.unwrap_or(true)));
    }
    if order_fields.is_empty() {
        return Ok(());
    }
    rows.sort_by(|a, b| {
        for (field, asc) in &order_fields {
            let av = a.1.get(field);
            let bv = b.1.get(field);
            let ord = match (av, bv) {
                (Some(x), Some(y)) => json_cmp(x, y).unwrap_or(std::cmp::Ordering::Equal),
                (Some(_), None) => std::cmp::Ordering::Greater,
                (None, Some(_)) => std::cmp::Ordering::Less,
                (None, None) => std::cmp::Ordering::Equal,
            };
            if ord != std::cmp::Ordering::Equal {
                return if *asc { ord } else { ord.reverse() };
            }
        }
        std::cmp::Ordering::Equal
    });
    Ok(())
}

struct DbPool {
    shards: Vec<Arc<PieskieoDb>>,
    template: PieskieoVectorParams,
}

impl DbPool {
    fn new(base_dir: &str, params: PieskieoVectorParams, shards: usize) -> anyhow::Result<Self> {
        let mut v = Vec::with_capacity(shards.max(1));
        for i in 0..shards.max(1) {
            let mut p = params.clone();
            p.shard_id = i;
            p.shard_total = shards.max(1);
            let dir = if shards > 1 {
                format!("{base_dir}/shard{i}")
            } else {
                base_dir.to_string()
            };
            std::fs::create_dir_all(&dir)?;
            v.push(Arc::new(PieskieoDb::open_with_params(&dir, p)?));
        }
        Ok(Self {
            shards: v,
            template: params,
        })
    }

    fn shard_for(&self, id: &Uuid) -> Arc<PieskieoDb> {
        if self.shards.len() == 1 {
            return self.shards[0].clone();
        }
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&id.as_bytes()[..8]);
        let idx = (u64::from_le_bytes(arr) as usize) % self.shards.len();
        self.shards[idx].clone()
    }

    fn each(&self) -> impl Iterator<Item = Arc<PieskieoDb>> + '_ {
        self.shards.iter().cloned()
    }

    fn counts(&self) -> HashMap<usize, usize> {
        let mut out = HashMap::new();
        for shard in &self.shards {
            let m = shard.metrics();
            out.insert(shard.shard_id(), m.docs + m.rows);
        }
        out
    }

    fn aggregate_metrics(&self) -> pieskieo_core::engine::MetricsSnapshot {
        let mut agg = pieskieo_core::engine::MetricsSnapshot {
            docs: 0,
            rows: 0,
            vectors: 0,
            vector_tombstones: 0,
            hnsw_ready: true,
            ef_search: 0,
            ef_construction: 0,
            wal_path: std::path::PathBuf::new(),
            wal_bytes: 0,
            snapshot_mtime: None,
            link_top_k: 0,
            shard_id: 0,
            shard_total: self.shards.len(),
        };
        for shard in &self.shards {
            let m = shard.metrics();
            agg.docs += m.docs;
            agg.rows += m.rows;
            agg.vectors += m.vectors;
            agg.vector_tombstones += m.vector_tombstones;
            agg.hnsw_ready &= m.hnsw_ready;
            agg.ef_search = m.ef_search;
            agg.ef_construction = m.ef_construction;
            agg.link_top_k = m.link_top_k;
            agg.wal_bytes += m.wal_bytes;
            agg.snapshot_mtime = agg.snapshot_mtime.or(m.snapshot_mtime);
        }
        agg
    }

    fn wal_all(&self) -> Vec<pieskieo_core::wal::RecordKind> {
        let mut out = Vec::new();
        for shard in &self.shards {
            if let Ok(mut r) = shard.wal_dump() {
                out.append(&mut r);
            }
        }
        out
    }

    fn template_params(&self) -> PieskieoVectorParams {
        self.template.clone()
    }
}

pub async fn serve() -> anyhow::Result<()> {
    let data_dir = std::env::var("PIESKIEO_DATA").unwrap_or_else(|_| default_data_dir());
    init_logging(&data_dir);

    let auth = Arc::new(RwLock::new(AuthConfig::from_env(&data_dir)));
    let params = vector_params_from_env();
    let shards = params.shard_total.max(1);
    let pool = Arc::new(RwLock::new(DbPool::new(&data_dir, params, shards)?));
    let limiter = Arc::new(RateLimiter::from_env());
    let audit = Arc::new(AuditLog::new(
        PathBuf::from(&data_dir).join("logs").join("audit.log"),
    ));

    let state = AppState {
        pool,
        auth,
        limiter,
        audit,
        data_dir,
        pause_writes: Arc::new(AtomicBool::new(false)),
        reshard_status: Arc::new(RwLock::new(None)),
    };

    // background WAL flusher (group commit) for better latency.
    let flush_ms = std::env::var("PIESKIEO_WAL_FLUSH_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(50);
    {
        let pool = state.pool.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(flush_ms));
            loop {
                interval.tick().await;
                let guard = pool.read().await;
                for shard in guard.each() {
                    if let Err(e) = shard.flush_wal() {
                        tracing::warn!("wal flush failed: {e}");
                    }
                }
            }
        });
    }

    if let Ok(secs) = std::env::var("PIESKIEO_SNAPSHOT_INTERVAL_SECS") {
        if let Ok(secs) = secs.parse::<u64>() {
            let pool = state.pool.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(secs));
                loop {
                    interval.tick().await;
                    let guard = pool.read().await;
                    for shard in guard.each() {
                        if let Err(e) = shard.save_vector_snapshot() {
                            tracing::warn!("snapshot save failed: {e}");
                        }
                    }
                }
            });
        }
    }

    if let Ok(secs) = std::env::var("PIESKIEO_REBUILD_INTERVAL_SECS") {
        if let Ok(secs) = secs.parse::<u64>() {
            let pool = state.pool.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(std::time::Duration::from_secs(secs));
                loop {
                    interval.tick().await;
                    let guard = pool.read().await;
                    for shard in guard.each() {
                        if let Err(e) = shard.rebuild_vectors() {
                            tracing::warn!("vector rebuild failed: {e}");
                        }
                    }
                }
            });
        }
    }

    let app = Router::new()
        .route("/healthz", get(health))
        .route("/v1/doc", post(put_doc))
        .route("/v1/doc/:id", get(get_doc))
        .route("/v1/doc/:id", delete(delete_doc))
        .route("/v1/doc/query", post(query_docs))
        .route("/v1/row", post(put_row))
        .route("/v1/row/:id", get(get_row))
        .route("/v1/row/:id", delete(delete_row))
        .route("/v1/row/query", post(query_rows))
        .route("/v1/vector", post(put_vector))
        .route("/v1/vector/:id/meta", post(update_vector_meta))
        .route("/v1/vector/config", post(update_vector_config))
        .route("/v1/vector/:id/meta/delete", post(delete_vector_meta_keys))
        .route("/v1/vector/:id", get(get_vector))
        .route("/v1/vector/vacuum", post(vacuum_vectors))
        .route("/v1/shard/which/:id", get(which_shard))
        .route("/v1/vector/search", post(search_vector))
        .route("/v1/vector/rebuild", post(rebuild_vectors))
        .route("/v1/vector/snapshot/save", post(save_snapshot))
        .route("/v1/vector/bulk", post(put_vector_bulk))
        .route("/v1/vector/:id", delete(delete_vector))
        .route("/v1/schema", post(set_schema))
        .route("/v1/sql", post(query_sql))
        .route("/v1/replica/wal", get(replica_wal))
        .route("/v1/replica/stream", get(replica_stream))
        .route("/v1/replica/apply", post(replica_apply))
        .route("/v1/replica/ws", get(replica_ws))
        .route("/metrics", get(metrics))
        .route("/v1/admin/reshard", post(reshard))
        .route("/v1/admin/reshard/status", get(reshard_status))
        .route("/v1/graph/edge", post(add_edge))
        .route("/v1/graph/:id", get(list_neighbors))
        .route("/v1/graph/:id/bfs", get(list_bfs))
        .route("/v1/graph/:id/dfs", get(list_dfs))
        .route("/v1/auth/users", get(list_users))
        .route("/v1/auth/users", post(create_user))
        .layer(middleware::from_fn_with_state(
            state.audit.clone(),
            audit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.limiter.clone(),
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(
            state.auth.clone(),
            auth_middleware,
        ))
        .layer(DefaultBodyLimit::max(
            std::env::var("PIESKIEO_BODY_LIMIT_MB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(10)
                * 1024
                * 1024,
        ))
        .with_state(state);

    let addr: SocketAddr = std::env::var("PIESKIEO_LISTEN")
        .unwrap_or_else(|_| "0.0.0.0:8000".into())
        .parse()?;

    let tls_files = (
        std::env::var("PIESKIEO_TLS_CERT").ok(),
        std::env::var("PIESKIEO_TLS_KEY").ok(),
    );

    if let (Some(cert), Some(key)) = tls_files {
        #[cfg(feature = "tls")]
        {
            let config = load_rustls_config(&cert, &key)?;
            tracing::info!(%addr, cert=%cert, key=%key, "listening with TLS");
            axum_server::bind_rustls(addr, config)
                .serve(app.into_make_service())
                .await?;
            return Ok(());
        }
        #[cfg(not(feature = "tls"))]
        {
            let _ = (cert, key);
            tracing::warn!("TLS paths provided but pieskieo-server built without `tls` feature; falling back to plaintext");
        }
    }

    tracing::info!(%addr, "listening (plaintext)");
    let listener = TcpListener::bind(addr).await?;

    // Graceful shutdown handler
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

    // Spawn signal handler
    tokio::spawn(async move {
        let ctrl_c = async {
            tokio::signal::ctrl_c()
                .await
                .expect("failed to install Ctrl+C handler");
        };

        #[cfg(unix)]
        let terminate = async {
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("failed to install SIGTERM handler")
                .recv()
                .await;
        };

        #[cfg(not(unix))]
        let terminate = std::future::pending::<()>();

        tokio::select! {
            _ = ctrl_c => {
                tracing::info!("received Ctrl+C, initiating shutdown");
            },
            _ = terminate => {
                tracing::info!("received SIGTERM, initiating shutdown");
            },
        }

        let _ = shutdown_tx.send(());
    });

    // Start server with graceful shutdown
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        shutdown_rx.await.ok();
        tracing::info!("shutting down gracefully...");
    })
    .await?;

    Ok(())
}

fn vector_params_from_env() -> PieskieoVectorParams {
    let metric = match std::env::var("PIESKIEO_VECTOR_METRIC")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "cosine" => pieskieo_core::vector::VectorMetric::Cosine,
        "dot" => pieskieo_core::vector::VectorMetric::Dot,
        _ => pieskieo_core::vector::VectorMetric::L2,
    };
    let ef_c = std::env::var("PIESKIEO_EF_CONSTRUCTION")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(200);
    let ef_s = std::env::var("PIESKIEO_EF_SEARCH")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(50);
    let max_el = std::env::var("PIESKIEO_VEC_MAX_ELEMENTS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100_000);
    let link_top_k = std::env::var("PIESKIEO_LINK_K")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(4);
    let shard_total = std::env::var("PIESKIEO_SHARD_TOTAL")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1);
    let shard_id = std::env::var("PIESKIEO_SHARD_ID")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);

    PieskieoVectorParams {
        metric,
        ef_construction: ef_c,
        ef_search: ef_s,
        max_elements: max_el,
        link_top_k,
        shard_id,
        shard_total,
    }
}

fn default_data_dir() -> String {
    #[cfg(target_os = "windows")]
    {
        std::env::var("APPDATA")
            .map(|p| format!("{p}\\Pieskieo"))
            .unwrap_or_else(|_| "C:\\ProgramData\\Pieskieo".into())
    }
    #[cfg(not(target_os = "windows"))]
    {
        if let Ok(xdg) = std::env::var("XDG_DATA_HOME") {
            return format!("{xdg}/pieskieo");
        }
        if let Ok(home) = std::env::var("HOME") {
            return format!("{home}/.local/share/pieskieo");
        }
        "/var/lib/pieskieo".into()
    }
}

fn init_logging(data_dir: &str) {
    let mode = std::env::var("PIESKIEO_LOG_MODE").unwrap_or_else(|_| "stdout".into());
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    let log_dir = std::env::var("PIESKIEO_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(data_dir).join("logs"));
    let file_appender = rolling::daily(log_dir, "pieskieo.log");
    let (file_writer, guard) = tracing_appender::non_blocking(file_appender);
    Box::leak(Box::new(guard));
    let writer: tracing_subscriber::fmt::writer::BoxMakeWriter = match mode.as_str() {
        "file" => tracing_subscriber::fmt::writer::BoxMakeWriter::new(
            file_writer.with_max_level(tracing::Level::TRACE),
        ),
        "both" => tracing_subscriber::fmt::writer::BoxMakeWriter::new(
            file_writer
                .and(std::io::stdout)
                .with_max_level(tracing::Level::TRACE),
        ),
        _ => tracing_subscriber::fmt::writer::BoxMakeWriter::new(
            std::io::stdout.with_max_level(tracing::Level::TRACE),
        ),
    };

    let layer = tracing_subscriber::fmt::layer()
        .with_writer(writer)
        .with_ansi(mode != "file");

    tracing_subscriber::registry()
        .with(filter)
        .with(layer)
        .init();
}

#[derive(Serialize)]
struct HealthStatus {
    status: String,
    version: String,
    uptime_seconds: u64,
    total_docs: usize,
    total_rows: usize,
    total_vectors: usize,
    shard_count: usize,
    auth_enabled: bool,
}

async fn health(State(state): State<AppState>) -> Result<Json<HealthStatus>, ApiError> {
    let guard = state.pool.read().await;
    let m = guard.aggregate_metrics();
    let auth_guard = state.auth.read().await;

    Ok(Json(HealthStatus {
        status: "healthy".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_seconds: 0, // TODO: track startup time
        total_docs: m.docs,
        total_rows: m.rows,
        total_vectors: m.vectors,
        shard_count: m.shard_total,
        auth_enabled: auth_guard.enabled(),
    }))
}

async fn put_doc(
    State(state): State<AppState>,
    Json(input): Json<DocInput>,
) -> Result<Json<ApiResponse<Uuid>>, ApiError> {
    if state.pause_writes.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(ApiError::Conflict("resharding in progress".into()));
    }
    let id = input.id.unwrap_or_else(Uuid::new_v4);
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .put_doc_ns(
            input.namespace.as_deref(),
            input.collection.as_deref(),
            id,
            input.data,
        )
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse { ok: true, data: id }))
}

async fn get_doc(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(ns): Query<NsParams>,
) -> Result<Json<ApiResponse<serde_json::Value>>, ApiError> {
    let doc = state
        .pool
        .read()
        .await
        .shard_for(&id)
        .get_doc_ns(ns.namespace.as_deref(), ns.collection.as_deref(), &id)
        .ok_or(ApiError::NotFound)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: doc,
    }))
}

async fn delete_doc(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(ns): Query<NsParams>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .delete_doc_ns(ns.namespace.as_deref(), ns.collection.as_deref(), &id)
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "deleted",
    }))
}

async fn put_row(
    State(state): State<AppState>,
    Json(input): Json<RowInput>,
) -> Result<Json<ApiResponse<Uuid>>, ApiError> {
    if state.pause_writes.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(ApiError::Conflict("resharding in progress".into()));
    }
    let id = input.id.unwrap_or_else(Uuid::new_v4);
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .put_row_ns(
            input.namespace.as_deref(),
            input.table.as_deref(),
            id,
            &input.data,
        )
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse { ok: true, data: id }))
}

async fn query_docs(
    State(state): State<AppState>,
    Json(input): Json<QueryInput>,
) -> Result<Json<ApiResponse<Vec<(Uuid, serde_json::Value)>>>, ApiError> {
    let mut hits = Vec::new();
    if let Some(sql) = input.sql {
        let guard = state.pool.read().await;
        for shard in guard.each() {
            match shard.query_sql(&sql)? {
                SqlResult::Select(mut rows) => {
                    hits.append(&mut rows);
                }
                _ => return Err(ApiError::BadRequest("SQL must be SELECT".into())),
            }
        }
        if let Some(limit) = input.limit {
            hits.truncate(limit);
        }
    } else {
        let guard = state.pool.read().await;
        for shard in guard.each() {
            hits.extend(shard.query_docs_ns(
                input.namespace.as_deref(),
                input.collection.as_deref(),
                &input.filter,
                usize::MAX,
                0,
            ));
        }
        let offset = input.offset.unwrap_or(0);
        hits = hits.into_iter().skip(offset).collect();
        if let Some(limit) = input.limit {
            hits.truncate(limit);
        }
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: hits,
    }))
}

async fn query_sql(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Json(input): Json<SqlInput>,
) -> Result<Json<ApiResponse<serde_json::Value>>, ApiError> {
    if !authorize(role, "/v1/sql", "POST") {
        return Err(ApiError::Forbidden);
    }
    let dialect = GenericDialect {};
    let ast = Parser::parse_sql(&dialect, &input.sql)
        .map_err(|e: sqlparser::parser::ParserError| ApiError::BadRequest(e.to_string()))?;
    if ast.is_empty() {
        return Err(ApiError::BadRequest("empty SQL".into()));
    }
    let first = &ast[0];
    let is_select = matches!(first, sqlparser::ast::Statement::Query(_));
    if is_select {
        let aggregate_plan = parse_simple_aggregate_plan(first)?;
        let fanout_sql = if let Some(plan) = aggregate_plan.as_ref() {
            build_raw_aggregate_fanout_sql(first, plan)?.unwrap_or_else(|| input.sql.clone())
        } else {
            strip_select_limit_offset(first)?.unwrap_or_else(|| input.sql.clone())
        };
        let mut rows = Vec::new();
        for shard in state.pool.read().await.each() {
            match shard.query_sql(&fanout_sql)? {
                SqlResult::Select(mut r) => rows.append(&mut r),
                _ => {}
            }
        }
        if let Some(plan) = aggregate_plan.as_ref() {
            rows = compute_grouped_aggregate_rows(rows, plan)?;
        }
        if let sqlparser::ast::Statement::Query(query) = first {
            sort_select_rows(&mut rows, query)?;
            let offset = query
                .offset
                .as_ref()
                .and_then(|o| match &o.value {
                    sqlparser::ast::Expr::Value(sqlparser::ast::Value::Number(n, _)) => {
                        n.parse::<usize>().ok()
                    }
                    _ => None,
                })
                .unwrap_or(0);
            rows = rows.into_iter().skip(offset).collect();
            let query_limit = query.limit.as_ref().and_then(|l| match l {
                sqlparser::ast::Expr::Value(sqlparser::ast::Value::Number(n, _)) => {
                    n.parse::<usize>().ok()
                }
                _ => None,
            });
            if let Some(limit) = query_limit {
                rows.truncate(limit);
            }
        }
        if let Some(limit) = input.limit {
            rows.truncate(limit);
        }
        return Ok(Json(ApiResponse {
            ok: true,
            data: serde_json::json!({ "kind": "select", "rows": rows }),
        }));
    }

    // non-select: route to first shard (or broadcast for update/delete)
    match first {
        sqlparser::ast::Statement::Update { .. } | sqlparser::ast::Statement::Delete { .. } => {
            let mut affected = 0usize;
            let guard = state.pool.read().await;
            for shard in guard.each() {
                match shard.query_sql(&input.sql)? {
                    SqlResult::Update { affected: a } | SqlResult::Delete { affected: a } => {
                        affected += a;
                    }
                    _ => {}
                }
            }
            Ok(Json(ApiResponse {
                ok: true,
                data: serde_json::json!({ "kind": "write", "affected": affected }),
            }))
        }
        sqlparser::ast::Statement::Insert { .. } => {
            let Some(id) = extract_insert_id(first)? else {
                return Err(ApiError::BadRequest(
                    "sharded /v1/sql INSERT requires an explicit UUID in id or _id".into(),
                ));
            };
            let shard = state.pool.read().await.shard_for(&id);
            match shard.query_sql(&input.sql)? {
                SqlResult::Insert { ids } => Ok(Json(ApiResponse {
                    ok: true,
                    data: serde_json::json!({ "kind": "insert", "ids": ids }),
                })),
                _ => Err(ApiError::Internal(anyhow::anyhow!("unexpected result"))),
            }
        }
        _ => Err(ApiError::BadRequest(
            "statement type not supported in /v1/sql".into(),
        )),
    }
}

async fn set_schema(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Json(input): Json<SchemaInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    if !authorize(role, "/v1/schema", "POST") {
        return Err(ApiError::Forbidden);
    }
    let def = SchemaDef {
        fields: input.fields,
    };
    match input.family.as_str() {
        "doc" | "docs" | "collection" | "collections" => {
            let guard = state.pool.read().await;
            for shard in guard.each() {
                shard.set_doc_schema(input.namespace.as_deref(), Some(&input.name), def.clone())?;
            }
        }
        "row" | "rows" | "table" | "tables" => {
            let guard = state.pool.read().await;
            for shard in guard.each() {
                shard.set_row_schema(input.namespace.as_deref(), Some(&input.name), def.clone())?;
            }
        }
        _ => return Err(ApiError::BadRequest("family must be doc or row".into())),
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "schema set",
    }))
}

async fn query_rows(
    State(state): State<AppState>,
    Json(input): Json<QueryInput>,
) -> Result<Json<ApiResponse<Vec<(Uuid, serde_json::Value)>>>, ApiError> {
    let mut hits = Vec::new();
    if let Some(sql) = input.sql {
        let guard = state.pool.read().await;
        for shard in guard.each() {
            match shard.query_sql(&sql)? {
                SqlResult::Select(mut rows) => {
                    hits.append(&mut rows);
                }
                _ => return Err(ApiError::BadRequest("SQL must be SELECT".into())),
            }
        }
        if let Some(limit) = input.limit {
            hits.truncate(limit);
        }
    } else {
        let guard = state.pool.read().await;
        for shard in guard.each() {
            hits.extend(shard.query_rows_ns(
                input.namespace.as_deref(),
                input.table.as_deref(),
                &input.filter,
                usize::MAX,
                0,
            ));
        }
        let offset = input.offset.unwrap_or(0);
        hits = hits.into_iter().skip(offset).collect();
        if let Some(limit) = input.limit {
            hits.truncate(limit);
        }
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: hits,
    }))
}

async fn get_row(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(ns): Query<NsParams>,
) -> Result<Json<ApiResponse<serde_json::Value>>, ApiError> {
    let row = state
        .pool
        .read()
        .await
        .shard_for(&id)
        .get_row_ns(ns.namespace.as_deref(), ns.table.as_deref(), &id)
        .ok_or(ApiError::NotFound)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: row,
    }))
}

async fn delete_row(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Query(ns): Query<NsParams>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .delete_row_ns(ns.namespace.as_deref(), ns.table.as_deref(), &id)
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "deleted",
    }))
}

async fn put_vector(
    State(state): State<AppState>,
    Json(input): Json<VectorInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    if state.pause_writes.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(ApiError::Conflict("resharding in progress".into()));
    }
    state
        .pool
        .read()
        .await
        .shard_for(&input.id)
        .put_vector_with_meta_ns(
            input.namespace.as_deref(),
            input.id,
            input.vector,
            input.meta,
        )
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "stored",
    }))
}

async fn put_vector_bulk(
    State(state): State<AppState>,
    Json(input): Json<VectorBulk>,
) -> Result<Json<ApiResponse<usize>>, ApiError> {
    if state.pause_writes.load(std::sync::atomic::Ordering::SeqCst) {
        return Err(ApiError::Conflict("resharding in progress".into()));
    }
    let mut stored = 0usize;
    let pool = state.pool.read().await;
    for item in input.items {
        pool.shard_for(&item.id)
            .put_vector_with_meta_ns(
                item.namespace.as_deref(),
                item.id,
                item.vector.clone(),
                item.meta.clone(),
            )
            .map_err(ApiError::from)?;
        stored += 1;
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: stored,
    }))
}

async fn delete_vector(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .delete_vector(&id)
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "deleted",
    }))
}

async fn get_vector(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<VectorOutput>>, ApiError> {
    let pool = state.pool.read().await;
    let (vector, meta) = pool
        .shard_for(&id)
        .get_vector(&id)
        .ok_or(ApiError::NotFound)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: VectorOutput { id, vector, meta },
    }))
}

async fn update_vector_meta(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(input): Json<VectorMetaInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .update_vector_meta(id, input.meta)
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "meta-updated",
    }))
}

async fn delete_vector_meta_keys(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    Json(input): Json<VectorMetaDeleteInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    state
        .pool
        .read()
        .await
        .shard_for(&id)
        .remove_vector_meta_keys(id, &input.keys)
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "meta-keys-removed",
    }))
}

async fn search_vector(
    State(state): State<AppState>,
    Json(input): Json<VectorSearchInput>,
) -> Result<Json<ApiResponse<Vec<pieskieo_core::VectorSearchResult>>>, ApiError> {
    let k = input.k.unwrap_or(10);
    let metric = match input.metric.as_deref() {
        Some("cosine") => pieskieo_core::vector::VectorMetric::Cosine,
        Some("dot") => pieskieo_core::vector::VectorMetric::Dot,
        Some("l2") => pieskieo_core::vector::VectorMetric::L2,
        _ => pieskieo_core::vector::VectorMetric::L2,
    };

    if let Some(ef) = input.ef_search {
        let guard = state.pool.read().await;
        for shard in guard.each() {
            shard.set_ef_search(ef);
        }
    }

    // For now metric selection is per-query; in future persist per-index config.
    let pool = state.pool.read().await;
    let futures = pool
        .each()
        .map(|shard| {
            let q = input.query.clone();
            let filter = input.filter_meta.clone();
            let ns = input.namespace.clone();
            tokio::task::spawn_blocking(move || match ns {
                Some(ref ns) => {
                    shard.search_vector_metric_ns(Some(ns.as_str()), &q, k, metric, filter)
                }
                None => shard.search_vector_metric(&q, k, metric, filter),
            })
        })
        .collect::<Vec<_>>();
    let mut all_hits = Vec::new();
    for res in join_all(futures).await {
        if let Ok(Ok(mut h)) = res {
            all_hits.append(&mut h);
        }
    }
    all_hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    all_hits.truncate(k);
    let mut hits = all_hits;
    if let Some(filter_ids) = input.filter_ids {
        let allow: std::collections::HashSet<Uuid> = filter_ids.into_iter().collect();
        hits.retain(|h| allow.contains(&h.id));
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: hits,
    }))
}

async fn update_vector_config(
    State(state): State<AppState>,
    Json(input): Json<VectorConfigInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    let pool = state.pool.read().await;
    if let Some(ef) = input.ef_search {
        for shard in pool.each() {
            shard.set_ef_search(ef);
        }
    }
    if let Some(efc) = input.ef_construction {
        for shard in pool.each() {
            shard.set_ef_construction(efc);
        }
    }
    if let Some(k) = input.link_top_k {
        for shard in pool.each() {
            if let Some(db_mut) = Arc::get_mut(&mut shard.clone()) {
                db_mut.set_link_top_k(k);
            } else {
                tracing::warn!("link_top_k update skipped (db state is shared)");
            }
        }
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "updated",
    }))
}

async fn rebuild_vectors(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    for shard in state.pool.read().await.each() {
        shard.rebuild_vectors().map_err(ApiError::from)?;
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "rebuilt",
    }))
}

async fn vacuum_vectors(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    for shard in state.pool.read().await.each() {
        shard.vacuum().map_err(ApiError::from)?;
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "vacuumed",
    }))
}

async fn save_snapshot(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    for shard in state.pool.read().await.each() {
        shard.save_vector_snapshot().map_err(ApiError::from)?;
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "saved",
    }))
}

async fn add_edge(
    State(state): State<AppState>,
    Json(input): Json<EdgeInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    let weight = input.weight.unwrap_or(1.0);
    state
        .pool
        .read()
        .await
        .shard_for(&input.src)
        .add_edge(input.src, input.dst, weight)
        .map_err(ApiError::from)?;
    Ok(Json(ApiResponse {
        ok: true,
        data: "stored",
    }))
}

async fn list_neighbors(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<Vec<pieskieo_core::Edge>>>, ApiError> {
    let edges = state.pool.read().await.shard_for(&id).neighbors(id, 100);
    Ok(Json(ApiResponse {
        ok: true,
        data: edges,
    }))
}

async fn list_bfs(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<Vec<pieskieo_core::Edge>>>, ApiError> {
    let edges = state.pool.read().await.shard_for(&id).bfs(id, 100);
    Ok(Json(ApiResponse {
        ok: true,
        data: edges,
    }))
}

async fn list_dfs(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<Vec<pieskieo_core::Edge>>>, ApiError> {
    let edges = state.pool.read().await.shard_for(&id).dfs(id, 100);
    Ok(Json(ApiResponse {
        ok: true,
        data: edges,
    }))
}

async fn which_shard(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
) -> Result<Json<ApiResponse<HashMap<&'static str, usize>>>, ApiError> {
    let pool = state.pool.read().await;
    let shard_total = pool.shards.len();
    let mut arr = [0u8; 8];
    arr.copy_from_slice(&id.as_bytes()[..8]);
    let shard_id = (u64::from_le_bytes(arr) as usize) % shard_total;
    let mut map = HashMap::new();
    map.insert("shard_id", shard_id);
    map.insert("shard_total", shard_total);
    Ok(Json(ApiResponse {
        ok: true,
        data: map,
    }))
}

async fn metrics(
    State(state): State<AppState>,
) -> Result<impl axum::response::IntoResponse, ApiError> {
    let guard = state.pool.read().await;
    let m = guard.aggregate_metrics();
    let mut body = format!(
        "pieskieo_docs {}\npieskieo_rows {}\npieskieo_vectors {}\npieskieo_vector_tombstones {}\npieskieo_hnsw_ready {}\npieskieo_ef_search {}\npieskieo_ef_construction {}\npieskieo_link_top_k {}\npieskieo_shard_total {}\n",
        m.docs,
        m.rows,
        m.vectors,
        m.vector_tombstones,
        m.hnsw_ready as u8,
        m.ef_search,
        m.ef_construction,
        m.link_top_k,
        m.shard_total,
    );
    let rejects = state.limiter.rejected.load(Ordering::Relaxed);
    body.push_str(&format!("pieskieo_rate_rejects {}\n", rejects));
    for (idx, shard) in guard.shards.iter().enumerate() {
        let s = shard.metrics();
        body.push_str(&format!(
            "pieskieo_shard_vectors{{shard=\"{}\"}} {}\npieskieo_shard_docs{{shard=\"{}\"}} {}\npieskieo_shard_rows{{shard=\"{}\"}} {}\n",
            idx, s.vectors, idx, s.docs, idx, s.rows
        ));
    }
    let resp = (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4",
        )],
        body,
    );
    Ok(resp)
}

async fn replica_wal(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Query(q): Query<WalQuery>,
) -> Result<Json<ApiResponse<WalExport>>, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    let since = q.since.unwrap_or(0);
    let mut slices = Vec::new();
    let guard = state.pool.read().await;
    for (idx, shard) in guard.shards.iter().enumerate() {
        let (records, end) = shard.wal_replay_since(since).map_err(ApiError::from)?;
        let mut encoded: Vec<String> = Vec::with_capacity(records.len());
        for rec in records {
            let bytes =
                bincode::serialize(&rec).map_err(|e| ApiError::Internal(anyhow::anyhow!(e)))?;
            encoded.push(B64.encode(bytes));
        }
        slices.push(WalShardSlice {
            shard: idx,
            end_offset: end,
            records: encoded,
        });
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: WalExport { slices },
    }))
}

async fn replica_apply(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Json(input): Json<ReplicationBatch>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    let mut records = Vec::new();
    for b64 in input.records {
        let bytes = B64
            .decode(b64)
            .map_err(|e| ApiError::BadRequest(format!("b64 decode error: {e}")))?;
        let rec: pieskieo_core::wal::RecordKind = bincode::deserialize(&bytes)
            .map_err(|e| ApiError::BadRequest(format!("decode: {e}")))?;
        records.push(rec);
    }
    let guard = state.pool.read().await;
    for shard in guard.each() {
        shard.apply_records(&records).map_err(ApiError::from)?;
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "applied",
    }))
}

// Long-polling incremental stream; returns immediately if new data exists, otherwise waits up to 10s.
async fn replica_stream(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Query(q): Query<WalQuery>,
) -> Result<Json<ApiResponse<WalStream>>, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    let since = q.since.unwrap_or(0);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    let mut last_offset = since;
    loop {
        let guard = state.pool.read().await;
        let mut slices = Vec::new();
        let mut max_end = since;
        for (idx, shard) in guard.shards.iter().enumerate() {
            let (records, end) = shard
                .wal_replay_since(last_offset)
                .map_err(ApiError::from)?;
            if !records.is_empty() {
                let mut encoded: Vec<String> = Vec::with_capacity(records.len());
                for rec in records {
                    let bytes = bincode::serialize(&rec)
                        .map_err(|e| ApiError::Internal(anyhow::anyhow!(e)))?;
                    encoded.push(B64.encode(bytes));
                }
                slices.push(WalShardSlice {
                    shard: idx,
                    end_offset: end,
                    records: encoded,
                });
            }
            if end > max_end {
                max_end = end;
            }
        }
        if !slices.is_empty() || max_end > since || std::time::Instant::now() >= deadline {
            return Ok(Json(ApiResponse {
                ok: true,
                data: WalStream {
                    end_offset: max_end,
                    slices,
                },
            }));
        }
        last_offset = max_end;
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
}

async fn replica_ws(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Query(q): Query<WalQuery>,
) -> Result<impl IntoResponse, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    let since = q.since.unwrap_or(0);
    Ok(ws.on_upgrade(move |socket| replica_ws_loop(socket, state, since)))
}

async fn replica_ws_loop(mut socket: WebSocket, state: AppState, since: u64) {
    let mut offset = since;
    loop {
        let guard = state.pool.read().await;
        let mut slices = Vec::new();
        let mut max_end = offset;
        for (idx, shard) in guard.shards.iter().enumerate() {
            match shard.wal_replay_since(offset) {
                Ok((records, end)) => {
                    if !records.is_empty() {
                        let mut encoded: Vec<String> = Vec::with_capacity(records.len());
                        for rec in records {
                            if let Ok(bytes) = bincode::serialize(&rec) {
                                encoded.push(B64.encode(bytes));
                            }
                        }
                        slices.push(WalShardSlice {
                            shard: idx,
                            end_offset: end,
                            records: encoded,
                        });
                    }
                    if end > max_end {
                        max_end = end;
                    }
                }
                Err(e) => {
                    let _ = socket.send(Message::Text(format!("error: {}", e))).await;
                }
            }
        }
        drop(guard);
        if !slices.is_empty() {
            let frame = serde_json::to_string(&WalStream {
                end_offset: max_end,
                slices,
            });
            if let Ok(f) = frame {
                if socket.send(Message::Text(f)).await.is_err() {
                    break;
                }
            }
            offset = max_end;
        } else {
            if socket.send(Message::Text("ping".into())).await.is_err() {
                break;
            }
        }
        if let Some(msg) = socket.recv().await {
            match msg {
                Ok(Message::Close(_)) => break,
                _ => {}
            }
        } else {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

async fn reshard(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Json(input): Json<ReshardRequest>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    state
        .pause_writes
        .store(true, std::sync::atomic::Ordering::SeqCst);
    {
        let mut s = state.reshard_status.write().await;
        *s = Some(ReshardReport {
            status: format!("reshard starting to {} shards", input.shards),
            paused: true,
            verified: false,
            before_counts: HashMap::new(),
            after_counts: HashMap::new(),
        });
    }
    let new_shards = input.shards.max(1);
    let (wal, template, before_counts) = {
        let guard = state.pool.read().await;
        (guard.wal_all(), guard.template_params(), guard.counts())
    };
    let mut params = template.clone();
    params.shard_total = new_shards;
    let new_pool = DbPool::new(&state.data_dir, params, new_shards).map_err(ApiError::Internal)?;
    for rec in &wal {
        for shard in new_pool.each() {
            shard
                .apply_records(&[rec.clone()])
                .map_err(ApiError::from)?;
        }
    }
    {
        let mut guard = state.pool.write().await;
        *guard = new_pool;
    }
    let after_counts = state.pool.read().await.counts();
    let verified = before_counts.values().sum::<usize>() == after_counts.values().sum::<usize>();
    state
        .pause_writes
        .store(false, std::sync::atomic::Ordering::SeqCst);
    {
        let mut s = state.reshard_status.write().await;
        *s = Some(ReshardReport {
            status: format!("reshard complete to {} shards", new_shards),
            paused: false,
            verified,
            before_counts,
            after_counts,
        });
    }
    Ok(Json(ApiResponse {
        ok: true,
        data: "resharded",
    }))
}

async fn reshard_status(
    State(state): State<AppState>,
) -> Result<Json<ApiResponse<ReshardStatus>>, ApiError> {
    let paused = state.pause_writes.load(std::sync::atomic::Ordering::SeqCst);
    let status = state.reshard_status.read().await.clone();
    Ok(Json(ApiResponse {
        ok: true,
        data: ReshardStatus { status, paused },
    }))
}

async fn auth_middleware(
    State(auth): State<Arc<RwLock<AuthConfig>>>,
    mut req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    let auth_guard = auth.read().await;
    if !auth_guard.enabled() {
        return Ok(next.run(req).await);
    }
    if let Some(header) = req.headers().get(axum::http::header::AUTHORIZATION) {
        if let Ok(val) = header.to_str() {
            if let Some(tok) = val.strip_prefix("Bearer ") {
                if let Some(expected) = &auth_guard.bearer {
                    if tok == expected {
                        req.extensions_mut().insert(Role::Admin);
                        return Ok(next.run(req).await);
                    }
                }
            }
            if let Some(basic) = val.strip_prefix("Basic ") {
                if let Ok(decoded) = B64.decode(basic) {
                    if let Ok(s) = String::from_utf8(decoded) {
                        if let Some((u, p)) = s.split_once(':') {
                            if auth_guard.check_lockout(u) {
                                return Err(ApiError::Unauthorized);
                            }
                            if let Some(user) = auth_guard.users.iter().find(|usr| {
                                usr.user == u && AuthConfig::verify_password(&usr.password_hash, p)
                            }) {
                                auth_guard.record_success(u);
                                req.extensions_mut().insert(user.role);
                                if authorize(user.role, req.uri().path(), req.method().as_str()) {
                                    return Ok(next.run(req).await);
                                } else {
                                    return Err(ApiError::Forbidden);
                                }
                            } else {
                                auth_guard.record_failure(u);
                                tracing::warn!(user = %u, "auth failure");
                            }
                        }
                    }
                }
            }
        }
    }
    Err(ApiError::Unauthorized)
}

async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    match limiter.allow(addr.ip()) {
        Ok(()) => Ok(next.run(req).await),
        Err(retry_after) => {
            let mut resp = axum::response::Response::new(axum::body::Body::empty());
            *resp.status_mut() = axum::http::StatusCode::TOO_MANY_REQUESTS;
            resp.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_str(&retry_after.as_secs().max(1).to_string())
                    .unwrap_or(axum::http::HeaderValue::from_static("60")),
            );
            Ok(resp)
        }
    }
}

async fn audit_middleware(
    State(audit): State<Arc<AuditLog>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request<Body>,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let start = Instant::now();
    let role = req.extensions().get::<Role>().copied();
    let res = next.run(req).await;
    let status = res.status().as_u16();
    audit.write(
        addr.ip(),
        method.as_str(),
        &path,
        status,
        role,
        start.elapsed(),
    );
    Ok(res)
}

fn authorize(role: Role, path: &str, method: &str) -> bool {
    use Role::*;
    if matches!(role, Admin) {
        return true;
    }
    let is_read = is_read_path(path, method);
    match role {
        Read => is_read,
        Write => is_read || is_write_path(path, method),
        Admin => true,
    }
}

fn is_read_path(path: &str, method: &str) -> bool {
    if path == "/healthz" || path == "/metrics" {
        return true;
    }
    let m = method.to_uppercase();
    if m == "GET" {
        return true;
    }
    // vector search is POST but read
    if path.contains("/vector/search") && m == "POST" {
        return true;
    }
    if path.contains("/graph") && m == "GET" {
        return true;
    }
    false
}

fn is_write_path(path: &str, method: &str) -> bool {
    let m = method.to_uppercase();
    if m == "POST" || m == "DELETE" || m == "PUT" || m == "PATCH" {
        return true;
    }
    // SQL endpoint treated as write by default
    if path.contains("/v1/sql") {
        return true;
    }
    false
}

#[cfg(feature = "tls")]
fn load_rustls_config(cert_path: &str, key_path: &str) -> anyhow::Result<RustlsConfig> {
    let certfile = File::open(cert_path)?;
    let mut reader = BufReader::new(certfile);
    let certs: Vec<Certificate> = certs(&mut reader)?.into_iter().map(Certificate).collect();

    let keyfile = File::open(key_path)?;
    let mut reader = BufReader::new(keyfile);
    let mut keys = Vec::new();
    for item in read_all(&mut reader)? {
        match item {
            Item::Pkcs8Key(key) | Item::RsaKey(key) => keys.push(PrivateKey(key)),
            _ => {}
        }
    }
    let key = keys
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no private key found in {}", key_path))?;

    Ok(RustlsConfig::from_der(certs, key)?)
}

#[derive(Debug)]
enum ApiError {
    NotFound,
    WrongShard,
    BadRequest(String),
    Conflict(String),
    Unauthorized,
    Forbidden,
    Internal(anyhow::Error),
}

impl From<PieskieoError> for ApiError {
    fn from(value: PieskieoError) -> Self {
        match value {
            PieskieoError::NotFound => ApiError::NotFound,
            PieskieoError::WrongShard => ApiError::WrongShard,
            PieskieoError::Validation(msg) => ApiError::BadRequest(msg),
            PieskieoError::UniqueViolation(field) => {
                ApiError::Conflict(format!("unique constraint on field {field}"))
            }
            other => ApiError::Internal(anyhow::Error::new(other)),
        }
    }
}

impl axum::response::IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        use axum::http::StatusCode;
        match self {
            ApiError::NotFound => StatusCode::NOT_FOUND.into_response(),
            ApiError::WrongShard => StatusCode::CONFLICT.into_response(),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg).into_response(),
            ApiError::Conflict(msg) => (StatusCode::CONFLICT, msg).into_response(),
            ApiError::Unauthorized => StatusCode::UNAUTHORIZED.into_response(),
            ApiError::Forbidden => StatusCode::FORBIDDEN.into_response(),
            ApiError::Internal(err) => {
                tracing::error!("api_error" = %err);
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        }
    }
}
async fn list_users(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
) -> Result<Json<ApiResponse<Vec<String>>>, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    let auth = state.auth.read().await;
    let users = auth
        .users
        .iter()
        .map(|u| format!("{} ({:?})", u.user, u.role))
        .collect();
    Ok(Json(ApiResponse {
        ok: true,
        data: users,
    }))
}

async fn create_user(
    State(state): State<AppState>,
    Extension(role): Extension<Role>,
    Json(input): Json<UserCreateInput>,
) -> Result<Json<ApiResponse<&'static str>>, ApiError> {
    if !matches!(role, Role::Admin) {
        return Err(ApiError::Forbidden);
    }
    if let Err(msg) = AuthConfig::validate_password(&input.pass) {
        return Err(ApiError::BadRequest(msg));
    }
    let mut auth = state.auth.write().await;
    let role = input
        .role
        .as_deref()
        .map(AuthConfig::parse_role)
        .unwrap_or(Role::Read);
    auth.users.push(UserRec {
        user: input.user,
        password_hash: AuthConfig::hash_password(&input.pass),
        role,
    });
    auth.persist();
    Ok(Json(ApiResponse {
        ok: true,
        data: "created",
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlparser::{dialect::GenericDialect, parser::Parser};

    #[test]
    fn extract_insert_id_reads_explicit_uuid() {
        let sql = "INSERT INTO docs.default.users (_id, name) VALUES ('550e8400-e29b-41d4-a716-446655440000', 'alice')";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, sql).expect("sql parses");
        let id = extract_insert_id(&ast[0]).expect("id extraction succeeds");
        assert_eq!(
            id,
            Some(Uuid::parse_str("550e8400-e29b-41d4-a716-446655440000").expect("uuid parses"),)
        );
    }

    #[test]
    fn extract_insert_id_returns_none_without_id_column() {
        let sql = "INSERT INTO docs.default.users (name) VALUES ('alice')";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, sql).expect("sql parses");
        let id = extract_insert_id(&ast[0]).expect("id extraction succeeds");
        assert_eq!(id, None);
    }

    #[test]
    fn strip_select_limit_offset_removes_query_bounds() {
        let sql = "SELECT name FROM docs.default.users ORDER BY age DESC LIMIT 2 OFFSET 1";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, sql).expect("sql parses");
        let rewritten = strip_select_limit_offset(&ast[0])
            .expect("rewrite succeeds")
            .expect("query rewrite exists");
        assert!(!rewritten.to_uppercase().contains(" LIMIT "));
        assert!(!rewritten.to_uppercase().contains(" OFFSET "));
        assert!(rewritten.to_uppercase().contains("ORDER BY"));
    }

    #[test]
    fn sort_select_rows_applies_global_order() {
        let sql = "SELECT name, age FROM docs.default.users ORDER BY age DESC";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, sql).expect("sql parses");
        let sqlparser::ast::Statement::Query(query) = &ast[0] else {
            panic!("expected query");
        };
        let mut rows = vec![
            (Uuid::nil(), serde_json::json!({"name": "alice", "age": 20})),
            (Uuid::nil(), serde_json::json!({"name": "bob", "age": 30})),
        ];
        sort_select_rows(&mut rows, query).expect("sort succeeds");
        assert_eq!(rows[0].1["name"], "bob");
    }

    #[test]
    fn compute_grouped_aggregate_rows_merges_count_sum_avg_min_max() {
        let specs = vec![
            AggregateSpec {
                alias: "c".to_string(),
                kind: AggregateKind::Count,
                field: None,
            },
            AggregateSpec {
                alias: "s".to_string(),
                kind: AggregateKind::Sum,
                field: Some("score".to_string()),
            },
            AggregateSpec {
                alias: "a".to_string(),
                kind: AggregateKind::Avg,
                field: Some("score".to_string()),
            },
            AggregateSpec {
                alias: "lo".to_string(),
                kind: AggregateKind::Min,
                field: Some("score".to_string()),
            },
            AggregateSpec {
                alias: "hi".to_string(),
                kind: AggregateKind::Max,
                field: Some("score".to_string()),
            },
        ];
        let rows = vec![
            (Uuid::nil(), serde_json::json!({"score": 5.0})),
            (Uuid::nil(), serde_json::json!({"score": 7.5})),
            (Uuid::nil(), serde_json::json!({"score": 12.5})),
        ];
        let plan = AggregatePlan {
            group_fields: Vec::new(),
            specs,
            having: None,
        };
        let merged = compute_grouped_aggregate_rows(rows, &plan).expect("merge succeeds");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].1["c"], 3);
        assert_eq!(merged[0].1["s"], 25.0);
        assert_eq!(merged[0].1["a"], 25.0 / 3.0);
        assert_eq!(merged[0].1["lo"], 5.0);
        assert_eq!(merged[0].1["hi"], 12.5);
    }

    #[test]
    fn build_raw_aggregate_fanout_sql_rewrites_avg_query() {
        let sql = "SELECT COUNT(*), AVG(age) AS avg_age, MAX(age) AS max_age FROM docs.default.users WHERE age > 10";
        let dialect = GenericDialect {};
        let ast = Parser::parse_sql(&dialect, sql).expect("sql parses");
        let plan = parse_simple_aggregate_plan(&ast[0])
            .expect("aggregate parsing succeeds")
            .expect("aggregate plan exists");
        let rewritten = build_raw_aggregate_fanout_sql(&ast[0], &plan)
            .expect("rewrite succeeds")
            .expect("query rewrite exists");
        let upper = rewritten.to_uppercase();
        assert!(upper.contains("SELECT _ID AS _ID, AGE AS AGE"));
        assert!(!upper.contains("COUNT("));
        assert!(!upper.contains("AVG("));
        assert!(!upper.contains("MAX("));
        assert!(upper.contains("WHERE AGE > 10"));
    }

    #[test]
    fn compute_grouped_aggregate_rows_groups_by_identifier_fields() {
        let plan = AggregatePlan {
            group_fields: vec![GroupField {
                source: "team".to_string(),
                alias: "team".to_string(),
            }],
            specs: vec![
                AggregateSpec {
                    alias: "c".to_string(),
                    kind: AggregateKind::Count,
                    field: None,
                },
                AggregateSpec {
                    alias: "avg_score".to_string(),
                    kind: AggregateKind::Avg,
                    field: Some("score".to_string()),
                },
            ],
            having: None,
        };
        let rows = vec![
            (
                Uuid::nil(),
                serde_json::json!({"team": "red", "score": 10.0}),
            ),
            (
                Uuid::nil(),
                serde_json::json!({"team": "red", "score": 20.0}),
            ),
            (
                Uuid::nil(),
                serde_json::json!({"team": "blue", "score": 9.0}),
            ),
        ];
        let mut merged = compute_grouped_aggregate_rows(rows, &plan).expect("group merge succeeds");
        merged.sort_by(|a, b| a.1["team"].as_str().cmp(&b.1["team"].as_str()));
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].1["team"], "blue");
        assert_eq!(merged[0].1["c"], 1);
        assert_eq!(merged[0].1["avg_score"], 9.0);
        assert_eq!(merged[1].1["team"], "red");
        assert_eq!(merged[1].1["c"], 2);
        assert_eq!(merged[1].1["avg_score"], 15.0);
    }

    #[test]
    fn compute_grouped_aggregate_rows_applies_having_filter() {
        let plan = AggregatePlan {
            group_fields: vec![GroupField {
                source: "team".to_string(),
                alias: "team".to_string(),
            }],
            specs: vec![AggregateSpec {
                alias: "c".to_string(),
                kind: AggregateKind::Count,
                field: None,
            }],
            having: Some(sqlparser::ast::Expr::BinaryOp {
                left: Box::new(sqlparser::ast::Expr::Identifier(
                    sqlparser::ast::Ident::new("c"),
                )),
                op: sqlparser::ast::BinaryOperator::Gt,
                right: Box::new(sqlparser::ast::Expr::Value(sqlparser::ast::Value::Number(
                    "1".to_string(),
                    false,
                ))),
            }),
        };
        let rows = vec![
            (Uuid::nil(), serde_json::json!({"team": "red"})),
            (Uuid::nil(), serde_json::json!({"team": "red"})),
            (Uuid::nil(), serde_json::json!({"team": "blue"})),
        ];
        let merged = compute_grouped_aggregate_rows(rows, &plan).expect("having merge succeeds");
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].1["team"], "red");
        assert_eq!(merged[0].1["c"], 2);
    }

    #[test]
    fn auth_config_generates_and_persists_default_admin() {
        let tmp = tempfile::tempdir().unwrap();
        let data_dir = tmp.path().to_str().unwrap();

        // Ensure no environment variables interfere
        std::env::remove_var("PIESKIEO_USERS");
        std::env::remove_var("PIESKIEO_AUTH_USER");
        std::env::remove_var("PIESKIEO_AUTH_PASSWORD");
        std::env::remove_var("PIESKIEO_TOKEN");

        let auth = AuthConfig::from_env(data_dir);

        assert_eq!(auth.users.len(), 1);
        assert_eq!(auth.users[0].user, "admin");
        assert_eq!(auth.users[0].role, Role::Admin);

        // Verify persistence
        let path = tmp.path().join("auth_users.json");
        assert!(path.exists());
        let content = std::fs::read_to_string(path).unwrap();
        let disk_users: Vec<UserDisk> = serde_json::from_str(&content).unwrap();
        assert_eq!(disk_users.len(), 1);
        assert_eq!(disk_users[0].user, "admin");
        assert_eq!(disk_users[0].role, "admin");

        // Verify the hash is valid
        // We don't have the plain password here, but we know it should be a hash.
        assert!(disk_users[0].pass.starts_with("$argon2id$"));
    }
}
