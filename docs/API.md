# Vantage API Reference

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [API Reference](#api-reference)
4. [Advanced Usage](#advanced-usage)
5. [Integration Examples](#integration-examples)

## Installation

```bash
# Install from source
git clone https://github.com/DHIWAHAR-K/Vantage.git
cd Vantage
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

**Requirements**:
- Python 3.9+
- MLX (automatically installed on Apple Silicon)
- macOS with Apple Silicon (M1/M2/M3)

## Quick Start

### Basic Usage

```python
from vantage import VantageAPI

# Load pretrained model
api = VantageAPI.from_pretrained("vantage-medium-8b")

# Define schema
schema = """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    salary REAL
);
"""

# Generate SQL
question = "What is the average salary by department?"
sql = api.generate(question, schema)

print(sql)
# Output: SELECT department, AVG(salary) FROM employees GROUP BY department
```

### Batch Processing

```python
questions = [
    "How many employees are there?",
    "Who earns the most?",
    "List all departments"
]

# Single schema for all questions
results = api.generate_batch(questions, schema)

for q, sql in zip(questions, results):
    print(f"Q: {q}")
    print(f"SQL: {sql}\n")
```

## API Reference

### VantageAPI

Main interface for text-to-SQL generation.

#### Constructor

```python
VantageAPI(
    model: VantageModel,
    tokenizer: Tokenizer,
    generation_config: Optional[GenerationConfig] = None
)
```

#### Class Methods

**from_pretrained**

```python
@classmethod
def from_pretrained(
    cls,
    model_path: str,
    generation_config: Optional[GenerationConfig] = None
) -> VantageAPI
```

Load model from checkpoint.

**Parameters**:
- `model_path` (str): Path to model directory
- `generation_config` (GenerationConfig, optional): Generation settings

**Returns**: VantageAPI instance

**Example**:
```python
api = VantageAPI.from_pretrained("./checkpoints/medium/best_model")
```

---

**from_config**

```python
@classmethod
def from_config(
    cls,
    config_path: str,
    generation_config: Optional[GenerationConfig] = None
) -> VantageAPI
```

Create model from configuration (randomly initialized).

**Parameters**:
- `config_path` (str): Path to YAML config file
- `generation_config` (GenerationConfig, optional): Generation settings

**Returns**: VantageAPI instance

---

#### Instance Methods

**generate**

```python
def generate(
    self,
    question: str,
    schema: Union[str, List[str]],
    return_metadata: bool = False
) -> Union[str, Dict]
```

Generate SQL query from natural language.

**Parameters**:
- `question` (str): Natural language question
- `schema` (str | List[str]): Database schema (CREATE TABLE statements or list)
- `return_metadata` (bool): Whether to return additional information

**Returns**:
- If `return_metadata=False`: SQL query string
- If `return_metadata=True`: Dictionary with SQL, question, schema, tables, columns

**Example**:
```python
# Simple usage
sql = api.generate("Count all users", schema)

# With metadata
result = api.generate("Count all users", schema, return_metadata=True)
print(result["sql"])
print(result["tables"])  # ['users']
print(result["columns"])  # {'users': ['id', 'name', 'email']}
```

---

**generate_batch**

```python
def generate_batch(
    self,
    questions: List[str],
    schemas: Union[List[str], str],
    return_metadata: bool = False
) -> Union[List[str], List[Dict]]
```

Generate SQL for multiple questions.

**Parameters**:
- `questions` (List[str]): List of questions
- `schemas` (List[str] | str): Schemas (one per question, or single for all)
- `return_metadata` (bool): Whether to return metadata

**Returns**: List of SQL queries or list of result dictionaries

**Example**:
```python
questions = ["Count users", "Average salary"]
results = api.generate_batch(questions, schema)
```

---

**validate_schema**

```python
def validate_schema(
    self,
    schema: str
) -> Dict[str, Any]
```

Validate and analyze database schema.

**Parameters**:
- `schema` (str): Schema text

**Returns**: Dictionary with schema analysis:
- `valid` (bool): Whether schema is valid
- `num_tables` (int): Number of tables
- `tables` (List[str]): Table names
- `columns` (Dict): Column names by table
- `total_columns` (int): Total number of columns

**Example**:
```python
analysis = api.validate_schema(schema)
print(f"Valid: {analysis['valid']}")
print(f"Tables: {analysis['tables']}")
```

---

**set_generation_config**

```python
def set_generation_config(self, **kwargs)
```

Update generation configuration.

**Parameters**: Any GenerationConfig attribute

**Example**:
```python
api.set_generation_config(
    temperature=0.7,
    num_beams=8,
    max_new_tokens=256
)
```

---

**__call__**

```python
def __call__(self, question: str, schema: str) -> str
```

Shorthand for `generate()`.

**Example**:
```python
sql = api("Count users", schema)  # Same as api.generate(...)
```

---

### GenerationConfig

Configuration for SQL generation.

```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 512          # Maximum SQL length
    temperature: float = 0.0            # Sampling temperature (0=greedy)
    top_p: float = 0.9                  # Nucleus sampling threshold
    top_k: int = 50                     # Top-k sampling
    num_beams: int = 4                  # Beam search width
    repetition_penalty: float = 1.0     # Penalize repetition
    length_penalty: float = 1.0         # Prefer longer/shorter outputs
    early_stopping: bool = True         # Stop when beam finished
    use_cache: bool = True              # Use KV cache for speed
    constrained_decoding: bool = True   # Only valid SQL tokens
    schema_aware: bool = True           # Only reference valid schema elements
```

**Example**:
```python
from vantage import GenerationConfig

config = GenerationConfig(
    num_beams=8,          # More beams = better quality, slower
    temperature=0.0,      # Greedy (deterministic) for SQL
    max_new_tokens=256,   # Shorter SQL queries
)

api = VantageAPI.from_pretrained("vantage-medium-8b", generation_config=config)
```

---

### Convenience Functions

**text_to_sql**

```python
def text_to_sql(
    question: str,
    schema: str,
    model_path: str = "vantage-medium-8b"
) -> str
```

One-off text-to-SQL generation (loads model each time).

**Example**:
```python
from vantage import text_to_sql

sql = text_to_sql(
    "Count users",
    "CREATE TABLE users (id INT, name TEXT);",
    model_path="./checkpoints/medium/best_model"
)
```

---

**create_api**

```python
def create_api(
    model_size: str = "medium",
    model_path: Optional[str] = None
) -> VantageAPI
```

Create API with defaults.

**Example**:
```python
from vantage import create_api

# Use pretrained medium model
api = create_api("medium")

# Or specify path
api = create_api(model_path="./my_model")
```

## Advanced Usage

### Schema Formats

**CREATE TABLE Statements**:
```python
schema = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    total REAL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
"""

sql = api.generate("User order totals", schema)
```

**Text Description**:
```python
schema = "Table users: id (int), name (text), email (text) | Table orders: id (int), user_id (int), total (real)"

sql = api.generate("User order totals", schema)
```

**Multiple Statements (List)**:
```python
schemas = [
    "CREATE TABLE users (id INT, name TEXT);",
    "CREATE TABLE products (id INT, name TEXT, price REAL);"
]

sql = api.generate("List all products", schemas)
```

### Custom Generation Settings

**Greedy Decoding** (fastest, deterministic):
```python
api.set_generation_config(
    num_beams=1,
    temperature=0.0
)
```

**Sampling** (diverse outputs):
```python
api.set_generation_config(
    num_beams=1,
    temperature=0.7,
    top_p=0.9,
    top_k=50
)
```

**High-Quality Beam Search** (best quality, slower):
```python
api.set_generation_config(
    num_beams=8,
    length_penalty=1.2,  # Prefer longer outputs
    early_stopping=True
)
```

### Error Handling

```python
try:
    sql = api.generate(question, schema)
except ValueError as e:
    print(f"Invalid input: {e}")
except RuntimeError as e:
    print(f"Generation failed: {e}")
```

### Streaming Generation

```python
# Not yet implemented
# Future API:
for token in api.generate_stream(question, schema):
    print(token, end='', flush=True)
```

## Integration Examples

### FastAPI Web Service

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from vantage import VantageAPI

app = FastAPI()
api = VantageAPI.from_pretrained("vantage-medium-8b")

class QueryRequest(BaseModel):
    question: str
    schema: str

class QueryResponse(BaseModel):
    sql: str
    valid: bool

@app.post("/generate", response_model=QueryResponse)
async def generate_sql(request: QueryRequest):
    try:
        sql = api.generate(request.question, request.schema)
        return QueryResponse(sql=sql, valid=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run: uvicorn api_server:app --host 0.0.0.0 --port 8000
```

**Usage**:
```bash
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{
        "question": "Count all users",
        "schema": "CREATE TABLE users (id INT, name TEXT);"
    }'
```

---

### Gradio Interface

```python
import gradio as gr
from vantage import VantageAPI

api = VantageAPI.from_pretrained("vantage-medium-8b")

def text_to_sql_interface(question, schema):
    try:
        sql = api.generate(question, schema)
        return sql
    except Exception as e:
        return f"Error: {str(e)}"

demo = gr.Interface(
    fn=text_to_sql_interface,
    inputs=[
        gr.Textbox(label="Question", placeholder="What is the average salary?"),
        gr.Textbox(label="Schema", placeholder="CREATE TABLE ...", lines=5)
    ],
    outputs=gr.Textbox(label="Generated SQL"),
    title="Vantage Text-to-SQL",
    description="Generate SQL queries from natural language"
)

demo.launch()
```

---

### Command-Line Tool

```python
#!/usr/bin/env python3
"""
vantage-cli: Command-line text-to-SQL tool
"""

import click
from vantage import VantageAPI

@click.command()
@click.option('--question', '-q', required=True, help='Natural language question')
@click.option('--schema', '-s', required=True, help='Database schema')
@click.option('--model', '-m', default='vantage-medium-8b', help='Model path')
def main(question, schema, model):
    """Generate SQL from natural language"""
    api = VantageAPI.from_pretrained(model)
    sql = api.generate(question, schema)
    click.echo(sql)

if __name__ == '__main__':
    main()
```

**Usage**:
```bash
chmod +x vantage-cli
./vantage-cli -q "Count users" -s "CREATE TABLE users (id INT);"
```

---

### Jupyter Notebook

```python
from vantage import VantageAPI
from IPython.display import display, Markdown

api = VantageAPI.from_pretrained("vantage-medium-8b")

def show_sql(question, schema):
    sql = api.generate(question, schema)
    display(Markdown(f"**Question**: {question}"))
    display(Markdown(f"```sql\n{sql}\n```"))

schema = "CREATE TABLE employees (id INT, name TEXT, salary REAL);"
show_sql("Who earns more than 100k?", schema)
```

---

### Database Integration

```python
import sqlite3
from vantage import VantageAPI

class SQLAssistant:
    def __init__(self, db_path, model_path="vantage-medium-8b"):
        self.conn = sqlite3.connect(db_path)
        self.api = VantageAPI.from_pretrained(model_path)
        self.schema = self._extract_schema()
    
    def _extract_schema(self):
        """Extract schema from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
        statements = [row[0] for row in cursor.fetchall()]
        return "\n\n".join(statements)
    
    def ask(self, question):
        """Ask question in natural language, get results"""
        # Generate SQL
        sql = self.api.generate(question, self.schema)
        print(f"Generated SQL: {sql}\n")
        
        # Execute
        cursor = self.conn.cursor()
        cursor.execute(sql)
        
        # Return results
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        return columns, rows

# Usage
assistant = SQLAssistant("my_database.db")
columns, rows = assistant.ask("What are the top 5 users by order count?")

for row in rows:
    print(dict(zip(columns, row)))
```

## Best Practices

1. **Schema Quality**: Provide complete, well-formatted schemas
2. **Question Clarity**: Clear, unambiguous questions work best
3. **Model Selection**: Use appropriate model size for your needs
4. **Error Handling**: Always validate and sanitize generated SQL
5. **Caching**: Reuse API instance (don't reload model repeatedly)
6. **Batch Processing**: Use `generate_batch()` for multiple queries
7. **Monitoring**: Track generation latency and success rates

## Troubleshooting

### Model Not Found

```python
# Error: Model path doesn't exist
api = VantageAPI.from_pretrained("nonexistent/path")

# Solution: Check path or download model
from pathlib import Path
assert Path("./checkpoints/medium/best_model").exists()
```

### Out of Memory

```python
# Error: Model too large for available memory

# Solution: Use smaller model or reduce batch size
api = VantageAPI.from_pretrained("vantage-small-2b")  # Instead of medium
```

### Poor Quality Output

```python
# Try increasing beam search
api.set_generation_config(num_beams=8)

# Or try different temperature
api.set_generation_config(temperature=0.3)

# Or regenerate multiple times and pick best
candidates = [api.generate(question, schema) for _ in range(5)]
best_sql = select_best(candidates)  # Your selection logic
```

## Next Steps

- See `TRAINING.md` for model training
- See `EVALUATION.md` for benchmarking
- See `ARCHITECTURE.md` for technical details
- Try interactive examples in the `notebooks/` directory
