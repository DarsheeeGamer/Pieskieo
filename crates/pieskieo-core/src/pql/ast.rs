// PQL Abstract Syntax Tree
// Production-ready AST types for Pieskieo Query Language
// Supports: vector search, graph traversal, relational queries, document operations
// ZERO compromises - complete PQL 3.0 specification

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// QUERY [WITH ctes] collection ...
    Query {
        with: Vec<Cte>,
        source: SourceExpr,
        operations: Vec<Operation>,
    },

    /// INSERT INTO collection ...
    Insert {
        target: String,
        rows: Vec<Vec<(String, Expression)>>,
        on_conflict: Option<OnConflict>,
        returning: Option<Vec<SelectField>>,
    },

    /// UPDATE collection SET ...
    Update {
        target: String,
        assignments: Vec<(String, Expression)>,
        filter: Option<Condition>,
        returning: Option<Vec<SelectField>>,
        from_source: Option<String>,
    },

    /// DELETE FROM collection WHERE ...
    Delete {
        target: String,
        filter: Option<Condition>,
        returning: Option<Vec<SelectField>>,
    },

    /// CREATE NODE/EDGE/INDEX/TABLE/COLLECTION ...
    Create(CreateStatement),

    /// ALTER TABLE ...
    AlterTable {
        name: String,
        operations: Vec<AlterTableOperation>,
    },

    /// DROP INDEX ...
    DropIndex { name: String, on: Option<String> },

    /// DROP TABLE / DROP COLLECTION
    DropCollection {
        name: String,
        is_table: bool,
        cascade: bool,
    },

    /// EXPLAIN query
    Explain {
        analyze: bool,
        statement: Box<Statement>,
    },

    /// UNION / INTERSECT / EXCEPT
    SetOperation {
        op: SetOperator,
        all: bool,
        left: Box<Statement>,
        right: Box<Statement>,
    },

    /// CREATE VIEW
    CreateView {
        name: String,
        if_not_exists: bool,
        query: Box<Statement>,
    },

    /// DROP VIEW
    DropView { name: String, if_exists: bool },

    /// BEGIN TRANSACTION
    Begin,

    /// COMMIT
    Commit,

    /// ROLLBACK [TO savepoint]
    Rollback { to: Option<String> },

    /// SAVEPOINT name
    Savepoint { name: String },

    /// RELEASE SAVEPOINT name
    ReleaseSavepoint { name: String },

    /// REMOVE EDGE src -> dst
    RemoveEdge { src: Expression, dst: Expression },

    /// MERGE INTO target USING source ON condition
    Merge {
        target: String,
        using: Box<Statement>,
        on: Condition,
        when_matched: Option<MergeAction>,
        when_not_matched: Option<MergeAction>,
    },

    /// INSERT INTO target SELECT ...
    InsertSelect {
        target: String,
        source: Box<Statement>,
        on_conflict: Option<OnConflict>,
        returning: Option<Vec<SelectField>>,
    },

    /// ADD EDGE src -> dst [TYPE type] [WEIGHT weight]
    AddEdge {
        src: Expression,
        dst: Expression,
        edge_type: Option<Expression>,
        weight: Option<Expression>,
    },

    /// TRUNCATE collection
    Truncate { name: String, is_table: bool },

    /// SHOW target
    Show(ShowTarget),

    /// CREATE SEQUENCE
    CreateSequence {
        name: String,
        if_not_exists: bool,
        start: i64,
        increment: i64,
        min_value: Option<i64>,
        max_value: Option<i64>,
        cycle: bool,
    },

    /// DROP SEQUENCE
    DropSequence { name: String, if_exists: bool },

    /// COPY collection FROM 'path' [FORMAT format]
    CopyFrom {
        collection: String,
        path: String,
        format: CopyFormat,
        header: bool,
    },

    /// COPY collection TO 'path' [FORMAT format]
    CopyTo {
        collection: String,
        path: String,
        format: CopyFormat,
        header: bool,
    },
}

/// Common Table Expression (WITH clause)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cte {
    pub name: String,
    pub recursive: bool,
    pub statement: Box<Statement>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceExpr {
    Collection(String),
    CollectionAs {
        name: String,
        alias: String,
    },
    Cte(String),
    Subquery {
        statement: Box<Statement>,
        alias: Option<String>,
    },
    Values {
        rows: Vec<Vec<(String, Expression)>>,
        alias: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// WHERE condition
    Filter(Condition),

    /// DISTINCT
    Distinct,

    /// SIMILAR TO vector TOP k THRESHOLD t
    VectorSearch {
        query_vector: Expression,
        field: Option<String>,
        top_k: usize,
        threshold: Option<f64>,
        metric: Option<VectorMetric>,
    },

    /// HYBRID SEARCH
    HybridSearch {
        query: Expression,
        field: Option<String>,
        top_k: usize,
        alpha: f64,
    },

    /// TRAVERSE edges WHERE ... DEPTH min TO max
    Traverse {
        edge_type: Option<String>,
        edge_filter: Option<Condition>,
        min_depth: usize,
        max_depth: usize,
        direction: TraverseDirection,
        mode: TraverseMode,
    },

    /// PATH traversal
    Path {
        from: Expression,
        to: Expression,
        mode: PathMode,
        edge_type: Option<String>,
        max_depth: usize,
    },

    /// MATCH graph_pattern
    Match { pattern: GraphPattern },

    /// JOIN other ON condition
    Join {
        join_type: JoinType,
        source: Box<SourceExpr>,
        condition: Condition,
    },

    /// GROUP BY fields [WITH ROLLUP | CUBE]
    GroupBy {
        fields: Vec<Expression>,
        mode: GroupByMode,
    },

    /// HAVING condition
    Having(Condition),

    /// COMPUTE field = expression
    Compute {
        assignments: Vec<(String, Expression)>,
    },

    /// ORDER BY field [ASC|DESC]
    OrderBy {
        fields: Vec<(Expression, SortOrder)>,
    },

    /// LIMIT n [OFFSET m]
    Limit { count: usize, offset: Option<usize> },

    /// SELECT fields
    Select { fields: Vec<SelectField> },

    /// FULLTEXT SEARCH
    FulltextSearch {
        query: Expression,
        field: Option<String>,
        top_k: usize,
    },

    /// UNNEST array field
    Unnest {
        field: Expression,
        alias: Option<String>,
        index_field: Option<String>,
        preserve: bool,
    },

    /// PIVOT value_field ON pivot_field IN (...) AGGREGATE func
    Pivot {
        value_field: Expression,
        pivot_field: Expression,
        pivot_values: Vec<String>,
        aggregate: String,
    },

    /// QUALIFY window filter
    Qualify { condition: Condition },

    /// SAMPLE n rows
    Sample { count: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupByMode {
    Regular,
    Rollup,
    Cube,
}

impl Default for GroupByMode {
    fn default() -> Self {
        GroupByMode::Regular
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathMode {
    Shortest,
    AllSimple,
    Any,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Condition {
    /// field = value, field > value, etc.
    Comparison {
        op: ComparisonOp,
        left: Expression,
        right: Expression,
    },

    /// cond1 AND cond2
    And {
        left: Box<Condition>,
        right: Box<Condition>,
    },

    /// cond1 OR cond2
    Or {
        left: Box<Condition>,
        right: Box<Condition>,
    },

    /// NOT cond
    Not { condition: Box<Condition> },

    /// field IN (val1, val2, ...)
    In {
        field: Expression,
        values: Vec<Expression>,
    },

    /// field NOT IN (val1, val2, ...)
    NotIn {
        field: Expression,
        values: Vec<Expression>,
    },

    /// field BETWEEN low AND high
    Between {
        field: Expression,
        low: Expression,
        high: Expression,
    },

    /// field IS NULL
    IsNull { field: Expression },

    /// field IS NOT NULL
    IsNotNull { field: Expression },

    /// EXISTS (subquery)
    Exists { subquery: Box<Statement> },

    /// Boolean expression (e.g. function call returning bool)
    Expr(Expression),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,
    Like,
    NotLike,
    ILike,
    RegexMatch,
    Contains,
    NotContains,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    /// Literal value
    Literal(Literal),

    /// Field reference (a.b.c)
    FieldAccess(Vec<String>),

    /// Function call: func(args)
    FunctionCall { name: String, args: Vec<Expression> },

    /// Binary operation: a + b, a * b, etc.
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },

    /// Unary operation: -a, NOT a
    UnaryOp {
        op: UnaryOperator,
        operand: Box<Expression>,
    },

    /// Subquery
    Subquery(Box<Statement>),

    /// Array literal: [1, 2, 3]
    Array(Vec<Expression>),

    /// Object literal: {a: 1, b: 2}
    Object(Vec<(String, Expression)>),

    /// Parameter reference: @param_name
    Parameter(String),

    /// CASE WHEN ... THEN ... ELSE ... END
    CaseWhen {
        operand: Option<Box<Expression>>,
        branches: Vec<(Expression, Expression)>,
        else_expr: Option<Box<Expression>>,
    },

    /// Window function: func() OVER (...)
    WindowFunction {
        func: Box<Expression>,
        args: Vec<Expression>,
        partition_by: Vec<Expression>,
        order_by: Vec<(Expression, SortOrder)>,
    },

    /// Condition as expression
    Condition(Box<Condition>),

    /// Array subscript: expr[index]
    Subscript {
        expr: Box<Expression>,
        index: Box<Expression>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Uuid(Uuid),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    Concat,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    ShiftLeft,
    ShiftRight,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Negate,
    Not,
    BitwiseNot,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SelectField {
    /// *
    All,

    /// field
    Field(Expression),

    /// field AS alias
    Aliased { expr: Expression, alias: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SortOrder {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
    Cross,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorMetric {
    L2,
    Cosine,
    Dot,
    Hamming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraverseDirection {
    Outgoing,
    Incoming,
    Both,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraverseMode {
    All,      // All paths
    Shortest, // Shortest path only
    Any,      // Any single path
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphPattern {
    pub nodes: Vec<NodePattern>,
    pub edges: Vec<EdgePattern>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodePattern {
    pub alias: Option<String>,
    pub labels: Vec<String>,
    pub properties: Option<Condition>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgePattern {
    pub alias: Option<String>,
    pub edge_type: Option<String>,
    pub properties: Option<Condition>,
    pub source: String, // Node aliasn
    pub target: String, // Node alias
    pub direction: TraverseDirection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreateStatement {
    /// CREATE COLLECTION ...
    Collection {
        name: String,
        fields: Vec<PropertyDef>,
        constraints: Vec<Constraint>,
    },

    /// CREATE NODE TYPE ...
    NodeType {
        name: String,
        properties: Vec<PropertyDef>,
        constraints: Vec<Constraint>,
    },

    /// CREATE EDGE TYPE ...
    EdgeType {
        name: String,
        source_type: Option<String>,
        target_type: Option<String>,
        properties: Vec<PropertyDef>,
        constraints: Vec<Constraint>,
    },

    /// CREATE INDEX ...
    Index {
        name: String,
        on: String,
        fields: Vec<String>,
        index_type: IndexType,
    },

    /// CREATE TABLE ...
    Table {
        name: String,
        columns: Vec<ColumnDef>,
        constraints: Vec<Constraint>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyDef {
    pub name: String,
    pub data_type: DataType,
    pub required: bool,
    pub unique: bool,
    pub default: Option<Literal>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub default: Option<Literal>,
    pub primary_key: bool,
    pub unique: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DataType {
    String,
    Integer,
    Float,
    Boolean,
    Date,
    Timestamp,
    Uuid,
    Json,
    Vector(usize), // Vector with dimension
    Bytes,
    Serial,
    BigSerial,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// UNIQUE(fields)
    Unique {
        name: Option<String>,
        fields: Vec<String>,
    },

    /// CHECK(condition)
    Check {
        name: Option<String>,
        condition: Condition,
    },

    /// FOREIGN KEY(fields) REFERENCES table(fields)
    ForeignKey {
        name: Option<String>,
        fields: Vec<String>,
        ref_table: String,
        ref_fields: Vec<String>,
        on_delete: ReferentialAction,
        on_update: ReferentialAction,
    },

    /// PRIMARY KEY(fields)
    PrimaryKey {
        name: Option<String>,
        fields: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReferentialAction {
    NoAction,
    Restrict,
    Cascade,
    SetNull,
    SetDefault,
}

impl Default for ReferentialAction {
    fn default() -> Self {
        ReferentialAction::NoAction
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    HNSW,
    FullText,
}

/// ALTER TABLE operations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AlterTableOperation {
    AddColumn(ColumnDef),
    DropColumn { name: String },
    RenameColumn { from: String, to: String },
    AlterColumnType { name: String, data_type: DataType },
    SetDefault { name: String, default: Expression },
    DropDefault { name: String },
    AddConstraint(Constraint),
    DropConstraint { name: String },
}

/// SET operators for combining queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SetOperator {
    Union,
    Intersect,
    Except,
}

/// SHOW targets
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShowTarget {
    Collections,
    Tables,
    Indexes { on: String },
    Schema { of: String },
    Sequences,
    Views,
}

/// COPY format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CopyFormat {
    Csv,
    Json,
    Parquet,
}

impl Default for CopyFormat {
    fn default() -> Self {
        CopyFormat::Csv
    }
}

/// ON CONFLICT clause
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OnConflict {
    pub target: Option<Vec<String>>,
    pub action: ConflictAction,
}

/// ON CONFLICT action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConflictAction {
    DoNothing,
    DoUpdate {
        assignments: Vec<(String, Expression)>,
    },
}

/// MERGE action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MergeAction {
    Update {
        assignments: Vec<(String, Expression)>,
    },
    Insert {
        fields: Vec<(String, Expression)>,
    },
    Delete,
}
