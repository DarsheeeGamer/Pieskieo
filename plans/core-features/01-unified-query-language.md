# PQL 3.0 - Pieskieo Query Language (Official Specification)

**Feature ID**: `core-features/01-unified-query-language.md`
**Status**: ✅ Production-Ready Design
**Priority**: CRITICAL - Foundation for everything
**Created**: 2026-02-09
**Version**: 3.0

---

## **Mission Statement**

PQL (Pieskieo Query Language) is the **single, unified query language** for Pieskieo that eliminates the need to learn SQL, MongoDB query language, Cypher, or any other database-specific syntax. 

**One language. All data models. Zero context switching.**

---

## **Core Principles**

1. **Flat Syntax** - No indentation required, newlines optional for readability
2. **Unified Storage** - Every entity has properties, nested data, vectors, and connections simultaneously
3. **Keywords for Operations** - `QUERY`, `WHERE`, `SIMILAR`, `TRAVERSE`, `JOIN`, etc.
4. **AND/OR for Conditions** - Logical operators within filter clauses
5. **Order-Based Execution** - Operations execute in written order (user controls flow)
6. **Semicolon Termination** - All queries end with `;`
7. **Zero Ambiguity** - No mode switching, no "which database am I querying?"

---

## **1. QUERYING DATA**

### **Basic Query Structure**
```pql
QUERY source_name
WHERE conditions
[operations...]
[output...]
;
```

### **Simple Queries**
```pql
-- All records from collection
QUERY users;

-- Filtered
QUERY users WHERE age > 25;

-- Multiple conditions with AND/OR
QUERY users WHERE age > 25 AND city = "NYC" OR status = "premium";

-- Nested document access (unified storage - no special syntax needed)
QUERY users WHERE metadata.preferences.theme = "dark" AND metadata.settings.notifications = true;

-- Array membership
QUERY users WHERE "admin" IN roles AND city IN ["NYC", "SF", "LA"];
```

---

## **2. VECTOR OPERATIONS**

Vector operations work on the **same unified storage** - no separate vector database.

### **Similarity Search**
```pql
-- Basic similarity with threshold
QUERY documents
SIMILAR TO @query_vector THRESHOLD 0.8
LIMIT 10;

-- Pre-filter then vector search
QUERY documents
WHERE category = "tech" AND published = true
SIMILAR TO @query_vector THRESHOLD 0.7
LIMIT 20;

-- Using embedding function inline
QUERY memories
SIMILAR TO embed("What did we discuss about the project?") THRESHOLD 0.75
WHERE timestamp > @last_week
LIMIT 10;

-- Distance-based (alternative syntax)
QUERY products
WHERE DISTANCE(vector, @query_vector) < 0.5
LIMIT 10;
```

### **Hybrid Search (Vector + Keyword)**
```pql
-- Combine vector similarity with keyword matching
QUERY articles
HYBRID SEARCH vector=embed("AI databases") keywords="vector search graph" weights=[0.7, 0.3]
WHERE published = true
LIMIT 20;

-- Multi-vector search (text + image embeddings)
QUERY products
SIMILAR TO vectors={text: @text_vec, image: @img_vec} weights=[0.6, 0.4]
WHERE in_stock = true
LIMIT 10;
```

---

## **3. GRAPH OPERATIONS**

Graph operations work on **connections in unified storage** - no separate graph database.

### **Traversal**
```pql
-- Simple edge traversal
QUERY users
WHERE id = @start_user
TRAVERSE edges WHERE type = "friend" DEPTH 2;

-- Multi-type edge traversal
QUERY nodes
TRAVERSE edges WHERE type IN ["follows", "likes", "shares"] DEPTH 1 TO 3;

-- Bidirectional traversal
QUERY entities
TRAVERSE edges WHERE type = "related" DIRECTION BOTH DEPTH 5;

-- Conditional traversal
QUERY products
TRAVERSE edges WHERE type = "similar" AND weight > 0.5 DEPTH 1 TO 2;
```

### **Path Finding**
```pql
-- Shortest path between two nodes
QUERY nodes
PATH SHORTEST FROM @node_a TO @node_b THROUGH edges WHERE weight > 0;

-- All paths (bounded depth)
QUERY nodes
PATH ALL FROM @start TO @end DEPTH MAX 5;

-- Pattern matching (Cypher-style but unified)
QUERY users
MATCH (user)-[rel:KNOWS]->(friend)-[rel2:WORKS_AT]->(company)
WHERE user.age > 25 AND company.industry = "tech";

-- Multiple pattern matching
QUERY social_graph
MATCH (a)-[r1]->(b)-[r2]->(c)
WHERE r1.type = "friend" AND r2.type = "colleague";
```

### **Graph Algorithms**
```pql
-- PageRank
QUERY web_pages
COMPUTE pagerank = PAGERANK(edges, iterations=100);

-- Community detection (Louvain)
QUERY social_network
COMPUTE community = LOUVAIN(edges);

-- Centrality measures
QUERY network
COMPUTE betweenness = BETWEENNESS(edges), closeness = CLOSENESS(edges);

-- Connected components
QUERY graph
COMPUTE component = CONNECTED_COMPONENTS(edges);
```

---

## **4. JOINS (Relational Operations)**

```pql
-- Inner join
QUERY orders
JOIN customers ON orders.customer_id = customers.id
WHERE orders.total > 100;

-- Left outer join
QUERY users
LEFT JOIN profiles ON users.profile_id = profiles.id
WHERE users.active = true;

-- Multiple joins
QUERY orders
JOIN customers ON orders.customer_id = customers.id
JOIN products ON orders.product_id = products.id
WHERE orders.status = "shipped";

-- Self-join
QUERY employees AS e1
JOIN employees AS e2 ON e1.manager_id = e2.id
WHERE e2.department = "engineering";

-- Join with subquery
QUERY users
JOIN (QUERY orders WHERE total > 1000 GROUP BY customer_id COMPUTE total_spent = SUM(total)) AS big_spenders
ON users.id = big_spenders.customer_id;

-- Cross join
QUERY products
CROSS JOIN categories
WHERE products.category_id = categories.id;
```

---

## **5. AGGREGATIONS**

```pql
-- Group by with aggregates
QUERY sales
GROUP BY product_id, DATE(timestamp)
COMPUTE total = SUM(amount), count = COUNT(*), avg = AVG(price);

-- Having clause (filter after grouping)
QUERY sales
GROUP BY product_id
COMPUTE total = SUM(amount)
HAVING total > 10000;

-- Multiple aggregates
QUERY orders
GROUP BY customer_id
COMPUTE order_count = COUNT(*), total_spent = SUM(total), avg_order = AVG(total), max_order = MAX(total), min_order = MIN(total);

-- Collect into arrays
QUERY orders
GROUP BY customer_id
COMPUTE order_ids = COLLECT(id), product_names = COLLECT(DISTINCT product_name);

-- Window functions
QUERY sales
COMPUTE row_num = ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY timestamp), running_total = SUM(amount) OVER (PARTITION BY product_id ORDER BY timestamp);

-- Lead/Lag for time-series
QUERY stock_prices
COMPUTE prev_price = LAG(price, 1) OVER (ORDER BY timestamp), next_price = LEAD(price, 1) OVER (ORDER BY timestamp), price_change = price - prev_price;
```

---

## **6. TRANSFORMATIONS**

### **Computed Fields**
```pql
-- Add computed columns
QUERY users
COMPUTE full_name = CONCAT(first_name, " ", last_name), age_years = YEAR_DIFF(birth_date, NOW()), score = price * 0.8 + rating * 0.2;

-- Conditional computation
QUERY products
COMPUTE discount = CASE WHEN category = "electronics" THEN price * 0.1 WHEN category = "books" THEN price * 0.2 ELSE 0 END;

-- Vector embedding generation
QUERY documents
COMPUTE vector = embed(content)
WHERE vector IS NULL;
```

### **Array Operations**
```pql
-- Unwind arrays (flatten)
QUERY documents
UNWIND tags AS tag
WHERE tag STARTS WITH "ai-";

-- Array filters
QUERY users
WHERE "admin" IN roles AND tags CONTAINS "verified";

-- Array length
QUERY posts
WHERE LENGTH(comments) > 10;

-- Array slicing
QUERY users
COMPUTE top_3_tags = SLICE(tags, 0, 3);
```

### **Projections (Output Shaping)**
```pql
-- Select specific fields
QUERY users
WHERE age > 25
SELECT id, name, email;

-- Nested output structure
QUERY users
SELECT id, user_info = {name: name, email: email}, stats = {login_count: login_count, last_seen: last_seen};

-- Computed output fields
QUERY products
SELECT id, name, discounted_price = price * 0.9, category;

-- Rename fields
QUERY users
SELECT user_id = id, full_name = name, contact = email;
```

---

## **7. SORTING & PAGINATION**

```pql
-- Order by single column
QUERY products
ORDER BY price DESC;

-- Order by multiple columns
QUERY products
ORDER BY category ASC, price DESC, name ASC;

-- Limit and offset
QUERY users
ORDER BY created_at DESC
LIMIT 20 OFFSET 40;

-- Top-k pattern
QUERY products
WHERE category = "electronics"
ORDER BY rating DESC
LIMIT 10;

-- Pagination helper
QUERY articles
ORDER BY published_at DESC
LIMIT 50 OFFSET @page * 50;
```

---

## **8. REAL-WORLD EXAMPLES**

### **Example 1: AI-Powered Semantic Search with Graph Context**
```pql
QUERY memories
WHERE timestamp > @last_week AND user_id = @current_user
SIMILAR TO embed("project discussion") THRESHOLD 0.75
TRAVERSE edges WHERE type = "relates_to" DEPTH 1 TO 3
JOIN users ON memories.user_id = users.id
COMPUTE relevance_score = VECTOR_SCORE() * 0.6 + GRAPH_CENTRALITY() * 0.4
WHERE relevance_score > 0.5
ORDER BY relevance_score DESC
LIMIT 10
SELECT id, content, users.name AS author, relevance_score, related_count = COUNT(traversed_edges);
```

**What this does:**
1. Filters memories from last week for current user (relational)
2. Finds semantically similar content via vector search
3. Traverses graph to find related topics (graph)
4. Joins with user data (relational)
5. Computes hybrid relevance score (vector + graph)
6. Filters by score threshold
7. Returns top 10 with structured output

---

### **Example 2: E-Commerce Product Recommendations**
```pql
QUERY products
WHERE in_stock = true AND category IN @user_preferences
SIMILAR TO @viewed_product_vector THRESHOLD 0.6
TRAVERSE purchase_graph WHERE edge.co_purchase_count > 10 DEPTH 1 TO 2
WHERE NOT (id IN @user_purchased_ids)
JOIN price_tiers ON products.tier_id = price_tiers.id
COMPUTE similarity_score = VECTOR_SCORE(), graph_score = GRAPH_WEIGHT_SUM(), final_score = similarity_score * 0.5 + graph_score * 0.5
ORDER BY final_score DESC
LIMIT 20
SELECT id, name, price_tiers.price AS price, final_score, recommendation_reason = {similar_to: @viewed_product_name, bought_with: COLLECT(related_products.name)};
```

**What this does:**
1. Filters available products in user's preferred categories
2. Finds products with similar embeddings (vector)
3. Traverses "frequently bought together" graph
4. Excludes already purchased products
5. Joins pricing data (relational)
6. Computes hybrid recommendation score
7. Returns top 20 with explanations

---

### **Example 3: Social Network Friend Suggestions**
```pql
QUERY users
WHERE username = "alice"
TRAVERSE edges WHERE type = "friend" DEPTH 1
SIMILAR TO current_user.interests_vector THRESHOLD 0.7
WHERE metadata.active_last_30_days = true
JOIN posts ON users.id = posts.author_id
GROUP BY users.id, users.name
COMPUTE friend_count = COUNT(DISTINCT friend_id), post_count = COUNT(posts.id), avg_engagement = AVG(posts.likes + posts.comments)
HAVING avg_engagement > 100
ORDER BY avg_engagement DESC
LIMIT 10
SELECT users.id, users.name, friend_count, post_count, avg_engagement;
```

**What this does:**
1. Starts from user "alice"
2. Finds friends of friends (graph traversal)
3. Filters by similar interests (vector)
4. Only active users (document metadata)
5. Joins with posts (relational)
6. Aggregates engagement metrics
7. Returns top 10 highly engaged users

---

### **Example 4: Time-Series Anomaly Detection**
```pql
QUERY sensor_readings
WHERE timestamp BETWEEN @start_time AND @end_time
COMPUTE expected_pattern = AVG_VECTOR(normal_readings.vector), anomaly_score = DISTANCE(reading_vector, expected_pattern)
WHERE anomaly_score > 2.0
TRAVERSE sensor_network WHERE edge.type = "physically_adjacent" DEPTH 1
GROUP BY sensor_id, HOUR(timestamp)
COMPUTE anomaly_count = COUNT(*), max_deviation = MAX(anomaly_score), affected_neighbors = COUNT(DISTINCT adjacent_sensors)
WHERE anomaly_count > 3 OR affected_neighbors > 2
ORDER BY max_deviation DESC
SELECT sensor_id, HOUR(timestamp) AS hour, anomaly_count, max_deviation, affected_neighbors;
```

---

## **9. MUTATIONS**

### **Insert**
```pql
-- Single insert
INSERT INTO users {id: 1, name: "Alice", age: 30, vector: embed("software engineer"), metadata: {theme: "dark", notifications: true}};

-- Bulk insert
INSERT INTO users [
  {id: 1, name: "Alice", age: 30},
  {id: 2, name: "Bob", age: 25},
  {id: 3, name: "Charlie", age: 35}
];

-- Insert with graph connections
INSERT INTO users {id: 1, name: "Alice", age: 30}
CONNECT TO {id: 2} AS "friend" WITH {since: "2020-01-01"};

-- Insert with computed fields
INSERT INTO documents {id: 1, content: "AI and machine learning", vector: embed(content), created_at: NOW()};
```

### **Update**
```pql
-- Update with filter
UPDATE users SET age = 31, metadata.last_login = NOW() WHERE id = 1;

-- Update with computed values
UPDATE products SET discounted_price = price * 0.9 WHERE category = "electronics";

-- Update vector embeddings
UPDATE documents SET vector = embed(content) WHERE vector IS NULL;

-- Update nested fields
UPDATE users SET metadata.preferences.theme = "light" WHERE id = 5;

-- Conditional update
UPDATE products SET status = CASE WHEN stock > 0 THEN "available" ELSE "out_of_stock" END;
```

### **Delete**
```pql
-- Delete with filter
DELETE FROM users WHERE age < 18 OR last_login < @one_year_ago;

-- Delete edges (graph)
DELETE EDGES FROM users WHERE type = "friend" AND created_at < @old_date;

-- Delete with subquery
DELETE FROM orders WHERE customer_id IN (QUERY inactive_customers WHERE last_order < @one_year_ago SELECT id);

-- Cascade delete (delete entity and all connections)
DELETE FROM users WHERE id = 5 CASCADE;
```

### **Upsert**
```pql
-- Insert or update if exists
UPSERT INTO users {id: 1, name: "Alice", age: 30} ON CONFLICT id DO UPDATE SET age = 30, name = "Alice";

-- Upsert with increment
UPSERT INTO counters {key: "page_views", count: 1} ON CONFLICT key DO UPDATE SET count = count + 1;
```

---

## **10. SCHEMA OPERATIONS**

### **Create Collection**
```pql
-- Basic collection
CREATE COLLECTION users {
  id: INTEGER PRIMARY KEY,
  name: TEXT NOT NULL,
  age: INTEGER CHECK (age > 0 AND age < 150),
  email: TEXT UNIQUE,
  vector: VECTOR(1536),
  metadata: JSON,
  created_at: TIMESTAMP DEFAULT NOW()
};

-- With constraints
CREATE COLLECTION products {
  id: INTEGER PRIMARY KEY,
  name: TEXT NOT NULL,
  price: DECIMAL(10, 2) CHECK (price > 0),
  category: TEXT,
  vector: VECTOR(768),
  FOREIGN KEY (category) REFERENCES categories(id)
};
```

### **Create Index**
```pql
-- B-tree index
CREATE INDEX idx_users_age ON users(age);

-- Composite index
CREATE INDEX idx_users_name_age ON users(name, age);

-- Vector index (HNSW)
CREATE VECTOR INDEX idx_users_vector ON users(vector) WITH (m=16, ef_construction=200, metric="cosine");

-- Full-text index
CREATE FULLTEXT INDEX idx_documents_content ON documents(content);

-- Partial index
CREATE INDEX idx_active_users ON users(email) WHERE active = true;

-- Covering index
CREATE INDEX idx_users_email_covering ON users(email) INCLUDE (name, created_at);

-- Hash index
CREATE HASH INDEX idx_users_email ON users(email);

-- Bloom filter index
CREATE BLOOM INDEX idx_users_filters ON users(col1, col2, col3) WITH (length=1024);
```

### **Alter Collection**
```pql
-- Add column
ALTER COLLECTION users ADD COLUMN phone TEXT;

-- Drop column
ALTER COLLECTION users DROP COLUMN phone;

-- Modify column type
ALTER COLLECTION users MODIFY COLUMN age BIGINT;

-- Add constraint
ALTER COLLECTION users ADD CONSTRAINT check_age CHECK (age >= 18);

-- Add index
ALTER COLLECTION users ADD INDEX idx_email ON (email);
```

### **Drop**
```pql
-- Drop collection
DROP COLLECTION users;

-- Drop index
DROP INDEX idx_users_age;

-- Drop with cascade
DROP COLLECTION users CASCADE;
```

---

## **11. TRANSACTIONS**

```pql
-- Basic transaction
BEGIN TRANSACTION;
UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;
INSERT INTO transfers {from_id: 1, to_id: 2, amount: 100, timestamp: NOW()};
COMMIT;

-- Rollback on error
BEGIN TRANSACTION;
DELETE FROM users WHERE id = 5;
ROLLBACK;

-- Isolation level
BEGIN TRANSACTION ISOLATION LEVEL SERIALIZABLE;
UPDATE inventory SET quantity = quantity - 1 WHERE product_id = 10;
COMMIT;

-- Savepoints
BEGIN TRANSACTION;
INSERT INTO users {id: 1, name: "Alice"};
SAVEPOINT sp1;
INSERT INTO users {id: 2, name: "Bob"};
ROLLBACK TO sp1;
COMMIT;
```

---

## **12. EXPLAIN & ANALYSIS**

```pql
-- Explain query plan
EXPLAIN QUERY users WHERE age > 25 SIMILAR TO @vec THRESHOLD 0.8;

-- Analyze query execution with timing
EXPLAIN ANALYZE QUERY products WHERE category = "tech" ORDER BY price LIMIT 10;

-- Verbose plan
EXPLAIN VERBOSE QUERY orders JOIN customers ON orders.customer_id = customers.id WHERE orders.total > 100;
```

---

## **13. BUILT-IN FUNCTIONS**

### **Vector Functions**
- `embed(text)` - Generate embedding from text
- `VECTOR_SCORE()` - Get similarity score in current context
- `DISTANCE(v1, v2)` - Calculate distance between vectors
- `AVG_VECTOR(vectors)` - Average of multiple vectors
- `NORMALIZE(vector)` - Normalize vector to unit length

### **Graph Functions**
- `CONNECTED(node1, node2)` - Check if two nodes are connected
- `GRAPH_CENTRALITY()` - Get centrality score in traversal context
- `GRAPH_WEIGHT_SUM()` - Sum of edge weights in traversal
- `SHORTEST_PATH(from, to)` - Find shortest path
- `PAGERANK(edges, iterations)` - Calculate PageRank
- `LOUVAIN(edges)` - Louvain community detection
- `BETWEENNESS(edges)` - Betweenness centrality
- `CLOSENESS(edges)` - Closeness centrality
- `CONNECTED_COMPONENTS(edges)` - Find connected components

### **Aggregate Functions**
- `COUNT()` - Count rows
- `SUM(column)` - Sum values
- `AVG(column)` - Average
- `MIN(column)` - Minimum
- `MAX(column)` - Maximum
- `COLLECT(column)` - Collect into array
- `COLLECT(DISTINCT column)` - Collect unique values
- `STRING_AGG(column, delimiter)` - Concatenate strings

### **Window Functions**
- `ROW_NUMBER() OVER (...)` - Row number
- `RANK() OVER (...)` - Rank with gaps
- `DENSE_RANK() OVER (...)` - Rank without gaps
- `SUM() OVER (...)` - Cumulative sum
- `AVG() OVER (...)` - Moving average
- `LAG(column, offset)` - Previous row value
- `LEAD(column, offset)` - Next row value

### **String Functions**
- `CONCAT(s1, s2, ...)` - Concatenate strings
- `UPPER(s)` - Convert to uppercase
- `LOWER(s)` - Convert to lowercase
- `LENGTH(s)` - String length
- `SUBSTRING(s, start, len)` - Extract substring
- `TRIM(s)` - Remove whitespace
- `STARTS WITH`, `ENDS WITH`, `CONTAINS` - String matching
- `MATCHES` - Regex matching

### **Date/Time Functions**
- `NOW()` - Current timestamp
- `DATE(timestamp)` - Extract date
- `HOUR(timestamp)` - Extract hour
- `DAY(timestamp)` - Extract day
- `MONTH(timestamp)` - Extract month
- `YEAR(timestamp)` - Extract year
- `YEAR_DIFF(date1, date2)` - Years between dates
- `DATE_ADD(date, interval)` - Add time interval

### **Array Functions**
- `LENGTH(array)` - Array length
- `SLICE(array, start, end)` - Array slice
- `CONTAINS(array, value)` - Check membership
- `UNNEST(array)` - Flatten array
- `ARRAY_AGG(column)` - Aggregate into array

### **Utility Functions**
- `UUID()` - Generate UUID
- `HASH(value)` - Hash value
- `RANDOM()` - Random number
- `COALESCE(v1, v2, ...)` - First non-null value
- `NULLIF(v1, v2)` - NULL if equal
- `CASE WHEN ... THEN ... ELSE ... END` - Conditional expression

---

## **14. OPERATORS**

### **Comparison**
- `=` Equal
- `!=` or `<>` Not equal
- `<` Less than
- `<=` Less than or equal
- `>` Greater than
- `>=` Greater than or equal

### **Logical**
- `AND` Logical and
- `OR` Logical or
- `NOT` Logical not

### **Membership**
- `IN` Value in list
- `NOT IN` Value not in list
- `CONTAINS` Array contains value
- `ANY` Match any in array
- `ALL` Match all in array

### **Pattern Matching**
- `LIKE` SQL-style pattern matching
- `MATCHES` Regex matching
- `STARTS WITH` String prefix
- `ENDS WITH` String suffix

### **Null Handling**
- `IS NULL` Check null
- `IS NOT NULL` Check not null

---

## **15. RESERVED KEYWORDS**

```
QUERY, WHERE, SELECT, INSERT, UPDATE, DELETE, UPSERT
JOIN, LEFT, RIGHT, FULL, CROSS, ON
GROUP BY, HAVING, ORDER BY, LIMIT, OFFSET
SIMILAR, TO, THRESHOLD, TRAVERSE, DEPTH, MATCH, PATH
COMPUTE, UNWIND, AS
BEGIN, COMMIT, ROLLBACK, TRANSACTION
CREATE, ALTER, DROP, COLLECTION, INDEX
VECTOR, GRAPH, EDGES, NODES
AND, OR, NOT, IN, LIKE, MATCHES
CASE, WHEN, THEN, ELSE, END
```

---

## **16. SYNTAX RULES**

1. **Statements end with semicolon** - `;` is required
2. **Keywords are case-insensitive** - `QUERY` = `query` = `Query`
3. **Identifiers are case-sensitive** - `users` != `Users`
4. **Strings use double quotes** - `"hello world"`
5. **Single quotes for literals** - `'2024-01-01'`
6. **Comments** - `-- single line` or `/* multi-line */`
7. **Variables start with @** - `@user_id`, `@query_vector`
8. **Operators have precedence** - `AND` before `OR`, use parentheses for clarity
9. **Newlines are optional** - Format for readability
10. **No indentation required** - Flat syntax

---

## **17. COMPARISON TO OTHER LANGUAGES**

### **PQL vs SQL**
```sql
-- SQL (PostgreSQL)
SELECT u.name, COUNT(o.id) as order_count
FROM users u
LEFT JOIN orders o ON u.id = o.customer_id
WHERE u.age > 25
GROUP BY u.name
HAVING COUNT(o.id) > 5;
```

```pql
-- PQL (Same query)
QUERY users
LEFT JOIN orders ON users.id = orders.customer_id
WHERE age > 25
GROUP BY name
COMPUTE order_count = COUNT(orders.id)
HAVING order_count > 5
SELECT name, order_count;
```

### **PQL vs MongoDB**
```javascript
// MongoDB
db.products.aggregate([
  { $match: { category: "electronics", price: { $lt: 100 } } },
  { $lookup: { from: "reviews", localField: "_id", foreignField: "product_id", as: "reviews" } },
  { $addFields: { avg_rating: { $avg: "$reviews.rating" } } },
  { $sort: { avg_rating: -1 } },
  { $limit: 10 }
])
```

```pql
-- PQL (Same query)
QUERY products
WHERE category = "electronics" AND price < 100
JOIN reviews ON products.id = reviews.product_id
GROUP BY products.id
COMPUTE avg_rating = AVG(reviews.rating)
ORDER BY avg_rating DESC
LIMIT 10;
```

### **PQL vs Cypher**
```cypher
// Cypher (Neo4j)
MATCH (user:User)-[:FRIEND]->(friend:User)-[:LIVES_IN]->(city:City)
WHERE user.name = "Alice"
RETURN friend.name, city.name
```

```pql
-- PQL (Same query)
QUERY users
WHERE name = "Alice"
MATCH (user)-[rel:FRIEND]->(friend)-[rel2:LIVES_IN]->(city)
SELECT friend.name, city.name;
```

---

## **18. ERROR HANDLING**

```pql
-- Try-catch for error handling
BEGIN TRY
  UPDATE accounts SET balance = balance - 1000 WHERE id = 1;
  UPDATE accounts SET balance = balance + 1000 WHERE id = 2;
  COMMIT;
CATCH
  ROLLBACK;
  SELECT ERROR_MESSAGE();
END;
```

---

## **19. PERFORMANCE HINTS**

```pql
-- Force index usage
QUERY users USE INDEX idx_users_age WHERE age > 25;

-- Disable index
QUERY users NO INDEX WHERE name = "Alice";

-- Parallel execution hint
QUERY large_table PARALLEL 8 WHERE category = "tech";

-- Join hint
QUERY orders HASH JOIN customers ON orders.customer_id = customers.id;
```

---

## **20. IMPLEMENTATION ROADMAP**

### **Phase 1: Parser** ✅
- Lexer: Tokenize PQL syntax
- Parser: Build AST from tokens
- Validator: Type checking and semantic validation

### **Phase 2: Optimizer** ✅
- Predicate pushdown
- Join reordering
- Index selection
- Query plan generation

### **Phase 3: Executor** ✅
- Unified execution engine
- Vector operation handlers
- Graph traversal engine
- Join algorithms
- Aggregation pipeline

### **Phase 4: Advanced Features** 🔄
- Query plan caching
- Adaptive query optimization
- Parallel execution
- Distributed query execution

---

## **STATUS**

**Implementation Status**: ✅ Specification Complete - Ready for Implementation  
**Parser**: Planned in `core-features/02-pql-parser.md`  
**Executor**: Planned in `core-features/03-pql-executor.md`  
**Optimizer**: Planned in `core-features/04-pql-optimizer.md`  
**Test Coverage**: 100% syntax coverage required  
**Documentation**: Complete language reference  

---

**This is THE language. One syntax for ALL data models. ZERO compromises.** 🚀
