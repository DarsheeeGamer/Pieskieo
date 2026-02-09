// PQL Abstract Syntax Tree
// Production-ready AST types for Pieskieo Query Language
// Supports: vector search, graph traversal, relational queries, document operations
// ZERO compromises - complete PQL 3.0 specification

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// QUERY collection ...
    Query {
        source: SourceExpr,
        operations: Vec<Operation>,
    },

    /// INSERT INTO collection ...
    Insert {
        target: String,
        values: Vec<(String, Expression)>,
    },

    /// UPDATE collection SET ...
    Update {
        target: String,
        assignments: Vec<(String, Expression)>,
        filter: Option<Condition>,
    },

    /// DELETE FROM collection WHERE ...
    Delete {
        target: String,
        filter: Option<Condition>,
    },

    /// CREATE NODE/EDGE/INDEX/TABLE ...
    Create(CreateStatement),

    /// EXPLAIN query
    Explain {
        analyze: bool,
        statement: Box<Statement>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SourceExpr {
    Collection(String),
    CollectionAs { name: String, alias: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Operation {
    /// WHERE condition
    Filter(Condition),

    /// SIMILAR TO vector TOP k THRESHOLD t
    VectorSearch {
        query_vector: Expression,
        field: Option<String>, // Which vector field to search
        top_k: usize,
        threshold: Option<f64>,
        metric: Option<VectorMetric>,
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

    /// MATCH graph_pattern
    Match { pattern: GraphPattern },

    /// JOIN other ON condition
    Join {
        join_type: JoinType,
        source: Box<SourceExpr>,
        condition: Condition,
    },

    /// GROUP BY fields
    GroupBy { fields: Vec<Expression> },

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
    Contains,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Negate,
    Not,
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
    pub source: String, // Node alias
    pub target: String, // Node alias
    pub direction: TraverseDirection,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CreateStatement {
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
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// UNIQUE(fields)
    Unique(Vec<String>),

    /// CHECK(condition)
    Check(Condition),

    /// FOREIGN KEY(fields) REFERENCES table(fields)
    ForeignKey {
        fields: Vec<String>,
        ref_table: String,
        ref_fields: Vec<String>,
    },

    /// PRIMARY KEY(fields)
    PrimaryKey(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexType {
    BTree,
    Hash,
    HNSW,
    FullText,
}
