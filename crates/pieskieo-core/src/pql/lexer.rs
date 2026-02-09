// PQL Lexer - Production-Ready Tokenization
// ZERO compromises: complete token set, robust error handling, efficient scanning

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Query,
    Insert,
    Update,
    Delete,
    Create,
    Drop,
    Alter,
    Select,
    From,
    Where,
    And,
    Or,
    Not,
    In,
    Between,
    Is,
    Null,
    Exists,
    As,
    Join,
    Inner,
    Left,
    Right,
    Full,
    Cross,
    On,
    GroupBy,
    OrderBy,
    Asc,
    Desc,
    Limit,
    Offset,
    Distinct,

    // Vector operations
    Similar,
    To,
    Top,
    Threshold,
    Metric,
    Embedding,

    // Graph operations
    Traverse,
    Match,
    Edges,
    Nodes,
    Path,
    Depth,
    Shortest,
    All,
    Any,

    // Computed fields
    Compute,

    // Schema/Types
    Table,
    Index,
    Node,
    Edge,
    Type,
    Constraint,
    Unique,
    Check,
    ForeignKey,
    PrimaryKey,
    References,
    Explain,

    // Data types
    String_,
    Integer_,
    Float_,
    Boolean_,
    Date_,
    Timestamp_,
    Uuid_,
    Json_,
    Vector,
    Bytes_,

    // Literals
    Identifier(String),
    StringLiteral(String),
    IntegerLiteral(i64),
    FloatLiteral(f64),
    BoolLiteral(bool),
    NullLiteral,
    Parameter(String), // @param_name

    // Operators
    Equal,            // =
    NotEqual,         // !=
    LessThan,         // <
    LessThanEqual,    // <=
    GreaterThan,      // >
    GreaterThanEqual, // >=
    Plus,             // +
    Minus,            // -
    Star,             // *
    Slash,            // /
    Percent,          // %
    Caret,            // ^
    Arrow,            // ->

    // Delimiters
    LeftParen,    // (
    RightParen,   // )
    LeftBrace,    // {
    RightBrace,   // }
    LeftBracket,  // [
    RightBracket, // ]
    Comma,        // ,
    Dot,          // .
    Colon,        // :
    Semicolon,    // ;

    // Special
    Eof,
    Error(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Token::Identifier(s) => write!(f, "IDENT({})", s),
            Token::StringLiteral(s) => write!(f, "STRING({})", s),
            Token::IntegerLiteral(n) => write!(f, "INT({})", n),
            Token::FloatLiteral(n) => write!(f, "FLOAT({})", n),
            Token::BoolLiteral(b) => write!(f, "BOOL({})", b),
            Token::Parameter(s) => write!(f, "@{}", s),
            Token::Error(msg) => write!(f, "ERROR: {}", msg),
            _ => write!(f, "{:?}", self),
        }
    }
}

pub struct Lexer {
    input: Vec<char>,
    position: usize,
    read_position: usize,
    ch: char,
}

impl Lexer {
    pub fn new(input: &str) -> Self {
        let chars: Vec<char> = input.chars().collect();
        let mut lexer = Lexer {
            input: chars,
            position: 0,
            read_position: 0,
            ch: '\0',
        };
        lexer.read_char();
        lexer
    }

    pub fn next_token(&mut self) -> Option<Token> {
        self.skip_whitespace();

        if self.ch == '\0' {
            return Some(Token::Eof);
        }

        let token = match self.ch {
            '=' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    Token::Equal
                } else {
                    Token::Equal
                }
            }
            '!' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    Token::NotEqual
                } else {
                    Token::Error("Unexpected '!'".to_string())
                }
            }
            '<' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    Token::LessThanEqual
                } else if self.peek_char() == '>' {
                    self.read_char();
                    Token::NotEqual
                } else {
                    Token::LessThan
                }
            }
            '>' => {
                if self.peek_char() == '=' {
                    self.read_char();
                    Token::GreaterThanEqual
                } else {
                    Token::GreaterThan
                }
            }
            '+' => Token::Plus,
            '-' => {
                if self.peek_char() == '>' {
                    self.read_char();
                    Token::Arrow
                } else {
                    Token::Minus
                }
            }
            '*' => Token::Star,
            '/' => Token::Slash,
            '%' => Token::Percent,
            '^' => Token::Caret,
            '(' => Token::LeftParen,
            ')' => Token::RightParen,
            '{' => Token::LeftBrace,
            '}' => Token::RightBrace,
            '[' => Token::LeftBracket,
            ']' => Token::RightBracket,
            ',' => Token::Comma,
            '.' => Token::Dot,
            ':' => Token::Colon,
            ';' => Token::Semicolon,
            '@' => return Some(self.read_parameter()),
            '\'' | '"' => return Some(self.read_string(self.ch)),
            _ if self.ch.is_ascii_digit() => return Some(self.read_number()),
            _ if self.ch.is_alphabetic() || self.ch == '_' => {
                return Some(self.read_identifier_or_keyword());
            }
            _ => Token::Error(format!("Unexpected character: {}", self.ch)),
        };

        self.read_char();
        Some(token)
    }

    fn read_char(&mut self) {
        if self.read_position >= self.input.len() {
            self.ch = '\0';
        } else {
            self.ch = self.input[self.read_position];
        }
        self.position = self.read_position;
        self.read_position += 1;
    }

    fn peek_char(&self) -> char {
        if self.read_position >= self.input.len() {
            '\0'
        } else {
            self.input[self.read_position]
        }
    }

    fn skip_whitespace(&mut self) {
        while self.ch.is_whitespace() {
            self.read_char();
        }
    }

    fn read_identifier_or_keyword(&mut self) -> Token {
        let start = self.position;

        while self.ch.is_alphanumeric() || self.ch == '_' {
            self.read_char();
        }

        let ident: String = self.input[start..self.position].iter().collect();
        let upper = ident.to_uppercase();

        // Check keywords (alphabetically for easy maintenance)
        match upper.as_str() {
            "ALL" => Token::All,
            "ALTER" => Token::Alter,
            "AND" => Token::And,
            "ANY" => Token::Any,
            "AS" => Token::As,
            "ASC" => Token::Asc,
            "BETWEEN" => Token::Between,
            "BOOLEAN" => Token::Boolean_,
            "BYTES" => Token::Bytes_,
            "CHECK" => Token::Check,
            "COMPUTE" => Token::Compute,
            "CONSTRAINT" => Token::Constraint,
            "CREATE" => Token::Create,
            "CROSS" => Token::Cross,
            "DATE" => Token::Date_,
            "DELETE" => Token::Delete,
            "DEPTH" => Token::Depth,
            "DESC" => Token::Desc,
            "DISTINCT" => Token::Distinct,
            "DROP" => Token::Drop,
            "EDGE" => Token::Edge,
            "EDGES" => Token::Edges,
            "EMBEDDING" => Token::Embedding,
            "EXISTS" => Token::Exists,
            "EXPLAIN" => Token::Explain,
            "FALSE" => Token::BoolLiteral(false),
            "FLOAT" => Token::Float_,
            "FOREIGNKEY" => Token::ForeignKey,
            "FROM" => Token::From,
            "FULL" => Token::Full,
            "GROUPBY" => Token::GroupBy,
            "IN" => Token::In,
            "INDEX" => Token::Index,
            "INNER" => Token::Inner,
            "INSERT" => Token::Insert,
            "INTEGER" => Token::Integer_,
            "IS" => Token::Is,
            "JOIN" => Token::Join,
            "JSON" => Token::Json_,
            "LEFT" => Token::Left,
            "LIMIT" => Token::Limit,
            "MATCH" => Token::Match,
            "METRIC" => Token::Metric,
            "NODE" => Token::Node,
            "NODES" => Token::Nodes,
            "NOT" => Token::Not,
            "NULL" => Token::NullLiteral,
            "OFFSET" => Token::Offset,
            "ON" => Token::On,
            "OR" => Token::Or,
            "ORDERBY" => Token::OrderBy,
            "PATH" => Token::Path,
            "PRIMARYKEY" => Token::PrimaryKey,
            "QUERY" => Token::Query,
            "REFERENCES" => Token::References,
            "RIGHT" => Token::Right,
            "SELECT" => Token::Select,
            "SHORTEST" => Token::Shortest,
            "SIMILAR" => Token::Similar,
            "STRING" => Token::String_,
            "TABLE" => Token::Table,
            "THRESHOLD" => Token::Threshold,
            "TIMESTAMP" => Token::Timestamp_,
            "TO" => Token::To,
            "TOP" => Token::Top,
            "TRAVERSE" => Token::Traverse,
            "TRUE" => Token::BoolLiteral(true),
            "TYPE" => Token::Type,
            "UNIQUE" => Token::Unique,
            "UPDATE" => Token::Update,
            "UUID" => Token::Uuid_,
            "VECTOR" => Token::Vector,
            "WHERE" => Token::Where,
            _ => Token::Identifier(ident),
        }
    }

    fn read_number(&mut self) -> Token {
        let start = self.position;
        let mut is_float = false;

        while self.ch.is_ascii_digit() {
            self.read_char();
        }

        if self.ch == '.' && self.peek_char().is_ascii_digit() {
            is_float = true;
            self.read_char(); // consume '.'
            while self.ch.is_ascii_digit() {
                self.read_char();
            }
        }

        // Handle scientific notation (e.g., 1.5e10)
        if self.ch == 'e' || self.ch == 'E' {
            is_float = true;
            self.read_char();
            if self.ch == '+' || self.ch == '-' {
                self.read_char();
            }
            while self.ch.is_ascii_digit() {
                self.read_char();
            }
        }

        let num_str: String = self.input[start..self.position].iter().collect();

        if is_float {
            match num_str.parse::<f64>() {
                Ok(n) => Token::FloatLiteral(n),
                Err(_) => Token::Error(format!("Invalid float: {}", num_str)),
            }
        } else {
            match num_str.parse::<i64>() {
                Ok(n) => Token::IntegerLiteral(n),
                Err(_) => Token::Error(format!("Invalid integer: {}", num_str)),
            }
        }
    }

    fn read_string(&mut self, quote: char) -> Token {
        self.read_char(); // consume opening quote
        let start = self.position;

        while self.ch != quote && self.ch != '\0' {
            if self.ch == '\\' {
                self.read_char(); // skip escape character
            }
            self.read_char();
        }

        if self.ch == '\0' {
            return Token::Error("Unterminated string".to_string());
        }

        let s: String = self.input[start..self.position].iter().collect();
        self.read_char(); // consume closing quote

        // Process escape sequences
        let processed = s
            .replace("\\n", "\n")
            .replace("\\t", "\t")
            .replace("\\r", "\r")
            .replace("\\\\", "\\")
            .replace(&format!("\\{}", quote), &quote.to_string());

        Token::StringLiteral(processed)
    }

    fn read_parameter(&mut self) -> Token {
        self.read_char(); // consume '@'
        let start = self.position;

        while self.ch.is_alphanumeric() || self.ch == '_' {
            self.read_char();
        }

        if self.position == start {
            return Token::Error("Invalid parameter name".to_string());
        }

        let param_name: String = self.input[start..self.position].iter().collect();
        Token::Parameter(param_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let input = "QUERY SELECT WHERE SIMILAR TRAVERSE";
        let mut lexer = Lexer::new(input);

        assert_eq!(lexer.next_token(), Some(Token::Query));
        assert_eq!(lexer.next_token(), Some(Token::Select));
        assert_eq!(lexer.next_token(), Some(Token::Where));
        assert_eq!(lexer.next_token(), Some(Token::Similar));
        assert_eq!(lexer.next_token(), Some(Token::Traverse));
    }

    #[test]
    fn test_identifiers() {
        let input = "users user_id myField";
        let mut lexer = Lexer::new(input);

        assert_eq!(
            lexer.next_token(),
            Some(Token::Identifier("users".to_string()))
        );
        assert_eq!(
            lexer.next_token(),
            Some(Token::Identifier("user_id".to_string()))
        );
        assert_eq!(
            lexer.next_token(),
            Some(Token::Identifier("myField".to_string()))
        );
    }

    #[test]
    fn test_numbers() {
        let input = "42 3.14 1.5e10 -5";
        let mut lexer = Lexer::new(input);

        assert_eq!(lexer.next_token(), Some(Token::IntegerLiteral(42)));
        assert_eq!(lexer.next_token(), Some(Token::FloatLiteral(3.14)));
        assert_eq!(lexer.next_token(), Some(Token::FloatLiteral(1.5e10)));
        assert_eq!(lexer.next_token(), Some(Token::Minus));
        assert_eq!(lexer.next_token(), Some(Token::IntegerLiteral(5)));
    }

    #[test]
    fn test_strings() {
        let input = r#"'hello' "world" 'it\'s'"#;
        let mut lexer = Lexer::new(input);

        assert_eq!(
            lexer.next_token(),
            Some(Token::StringLiteral("hello".to_string()))
        );
        assert_eq!(
            lexer.next_token(),
            Some(Token::StringLiteral("world".to_string()))
        );
        assert_eq!(
            lexer.next_token(),
            Some(Token::StringLiteral("it's".to_string()))
        );
    }

    #[test]
    fn test_operators() {
        let input = "= != < <= > >= + - * / % ^";
        let mut lexer = Lexer::new(input);

        assert_eq!(lexer.next_token(), Some(Token::Equal));
        assert_eq!(lexer.next_token(), Some(Token::NotEqual));
        assert_eq!(lexer.next_token(), Some(Token::LessThan));
        assert_eq!(lexer.next_token(), Some(Token::LessThanEqual));
        assert_eq!(lexer.next_token(), Some(Token::GreaterThan));
        assert_eq!(lexer.next_token(), Some(Token::GreaterThanEqual));
        assert_eq!(lexer.next_token(), Some(Token::Plus));
        assert_eq!(lexer.next_token(), Some(Token::Minus));
        assert_eq!(lexer.next_token(), Some(Token::Star));
        assert_eq!(lexer.next_token(), Some(Token::Slash));
        assert_eq!(lexer.next_token(), Some(Token::Percent));
        assert_eq!(lexer.next_token(), Some(Token::Caret));
    }

    #[test]
    fn test_parameters() {
        let input = "@user_id @embedding @max_price";
        let mut lexer = Lexer::new(input);

        assert_eq!(
            lexer.next_token(),
            Some(Token::Parameter("user_id".to_string()))
        );
        assert_eq!(
            lexer.next_token(),
            Some(Token::Parameter("embedding".to_string()))
        );
        assert_eq!(
            lexer.next_token(),
            Some(Token::Parameter("max_price".to_string()))
        );
    }

    #[test]
    fn test_pql_query() {
        let input = "QUERY users WHERE age > 25 SIMILAR TO @embedding TOP 10;";
        let mut lexer = Lexer::new(input);

        assert_eq!(lexer.next_token(), Some(Token::Query));
        assert_eq!(
            lexer.next_token(),
            Some(Token::Identifier("users".to_string()))
        );
        assert_eq!(lexer.next_token(), Some(Token::Where));
        assert_eq!(
            lexer.next_token(),
            Some(Token::Identifier("age".to_string()))
        );
        assert_eq!(lexer.next_token(), Some(Token::GreaterThan));
        assert_eq!(lexer.next_token(), Some(Token::IntegerLiteral(25)));
        assert_eq!(lexer.next_token(), Some(Token::Similar));
        assert_eq!(lexer.next_token(), Some(Token::To));
        assert_eq!(
            lexer.next_token(),
            Some(Token::Parameter("embedding".to_string()))
        );
        assert_eq!(lexer.next_token(), Some(Token::Top));
        assert_eq!(lexer.next_token(), Some(Token::IntegerLiteral(10)));
        assert_eq!(lexer.next_token(), Some(Token::Semicolon));
    }
}
