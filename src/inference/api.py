"""
Simple Python API for Vantage model inference
"""

from pathlib import Path
from typing import List, Dict, Optional, Union
import mlx.core as mx

from ..models.text2sql_model import VantageModel, VantageConfig
from ..data.preprocessing import Tokenizer
from ..data.schema_utils import SchemaParser, SchemaGraph
from .generator import SQLGenerator, GenerationConfig


class VantageAPI:
    """
    High-level API for Vantage text-to-SQL model.
    
    Example usage:
        ```python
        api = VantageAPI.from_pretrained("vantage-medium-8b")
        
        schema = '''
        CREATE TABLE users (
            id INT PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100)
        );
        '''
        
        question = "How many users are in the database?"
        sql = api.generate(question, schema)
        print(sql)  # SELECT COUNT(*) FROM users
        ```
    """
    
    def __init__(
        self,
        model: VantageModel,
        tokenizer: Tokenizer,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.generator = SQLGenerator(
            model=model,
            tokenizer=tokenizer,
            config=generation_config or GenerationConfig(),
        )
        
        # Schema parser for advanced features
        self.schema_parser = SchemaParser()
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "VantageAPI":
        """
        Load model from checkpoint.
        
        Args:
            model_path: Path to model checkpoint
            generation_config: Optional generation configuration
            
        Returns:
            VantageAPI instance
        """
        print(f"Loading model from {model_path}")
        
        model = VantageModel.from_pretrained(model_path)
        tokenizer = Tokenizer.from_pretrained(model_path)
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
    
    @classmethod
    def from_config(
        cls,
        config_path: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> "VantageAPI":
        """
        Create model from configuration (randomly initialized).
        
        Args:
            config_path: Path to configuration file
            generation_config: Optional generation configuration
            
        Returns:
            VantageAPI instance
        """
        model = VantageModel.from_config(config_path)
        tokenizer = Tokenizer()  # Default tokenizer
        
        return cls(
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config,
        )
    
    def generate(
        self,
        question: str,
        schema: Union[str, List[str]],
        return_metadata: bool = False,
    ) -> Union[str, Dict]:
        """
        Generate SQL query from natural language question.
        
        Args:
            question: Natural language question
            schema: Database schema (CREATE TABLE statements or text description)
            return_metadata: Whether to return additional metadata
            
        Returns:
            SQL query string, or dict with query and metadata
        """
        # Parse schema if needed
        if isinstance(schema, list):
            schema = "\n".join(schema)
        
        # Extract table/column information for schema-aware generation
        tables, columns = self._parse_schema(schema)
        
        # Generate SQL
        sql = self.generator.generate(
            question=question,
            schema=schema,
            tables=tables,
            columns=columns,
        )
        
        if return_metadata:
            return {
                "sql": sql,
                "question": question,
                "schema": schema,
                "tables": tables,
                "columns": columns,
            }
        else:
            return sql
    
    def generate_batch(
        self,
        questions: List[str],
        schemas: Union[List[str], str],
        return_metadata: bool = False,
    ) -> Union[List[str], List[Dict]]:
        """
        Generate SQL for batch of questions.
        
        Args:
            questions: List of natural language questions
            schemas: Schema(s) - single schema for all or list of schemas
            return_metadata: Whether to return additional metadata
            
        Returns:
            List of SQL queries or list of metadata dicts
        """
        # Handle single schema for all questions
        if isinstance(schemas, str):
            schemas = [schemas] * len(questions)
        
        results = []
        
        for question, schema in zip(questions, schemas):
            result = self.generate(
                question=question,
                schema=schema,
                return_metadata=return_metadata,
            )
            results.append(result)
        
        return results
    
    def _parse_schema(self, schema: str) -> tuple:
        """
        Parse schema to extract tables and columns.
        
        Args:
            schema: Schema text
            
        Returns:
            (tables, columns) tuple
        """
        tables = []
        columns = {}
        
        try:
            # Try parsing as CREATE TABLE statements
            statements = schema.split(";")
            
            for stmt in statements:
                stmt = stmt.strip()
                if not stmt or not stmt.upper().startswith("CREATE TABLE"):
                    continue
                
                table = self.schema_parser.parse_create_table(stmt)
                tables.append(table.name)
                columns[table.name] = [col.name for col in table.columns]
        
        except:
            # Fallback to text parsing
            parsed_tables = self.schema_parser.parse_schema_text(schema)
            
            for table_name, table in parsed_tables.items():
                tables.append(table_name)
                columns[table_name] = [col.name for col in table.columns]
        
        return tables, columns
    
    def validate_schema(self, schema: str) -> Dict[str, any]:
        """
        Validate and analyze database schema.
        
        Args:
            schema: Schema text
            
        Returns:
            Dictionary with schema analysis
        """
        tables, columns = self._parse_schema(schema)
        
        # Build schema graph
        schema_graph = None
        try:
            statements = [s.strip() for s in schema.split(";") if "CREATE TABLE" in s.upper()]
            self.schema_parser.build_from_create_statements(statements)
            schema_graph = SchemaGraph(self.schema_parser.tables)
        except:
            pass
        
        analysis = {
            "valid": len(tables) > 0,
            "num_tables": len(tables),
            "tables": tables,
            "columns": columns,
            "total_columns": sum(len(cols) for cols in columns.values()),
        }
        
        if schema_graph:
            analysis["has_relationships"] = len(schema_graph.adjacency) > 0
        
        return analysis
    
    def set_generation_config(self, **kwargs):
        """
        Update generation configuration.
        
        Args:
            **kwargs: Generation config parameters
        """
        for key, value in kwargs.items():
            if hasattr(self.generator.config, key):
                setattr(self.generator.config, key, value)
    
    def __call__(self, question: str, schema: str) -> str:
        """
        Shorthand for generate().
        
        Args:
            question: Natural language question
            schema: Database schema
            
        Returns:
            Generated SQL query
        """
        return self.generate(question, schema)


# Convenience functions

def text_to_sql(
    question: str,
    schema: str,
    model_path: str = "vantage-medium-8b",
) -> str:
    """
    Convenience function for one-off text-to-SQL generation.
    
    Args:
        question: Natural language question
        schema: Database schema
        model_path: Path to model checkpoint
        
    Returns:
        Generated SQL query
    """
    api = VantageAPI.from_pretrained(model_path)
    return api.generate(question, schema)


def create_api(
    model_size: str = "medium",
    model_path: Optional[str] = None,
) -> VantageAPI:
    """
    Create API instance with default configuration.
    
    Args:
        model_size: Model size ("small", "medium", "large")
        model_path: Optional path to checkpoint
        
    Returns:
        VantageAPI instance
    """
    if model_path is None:
        model_path = f"vantage-{model_size}"
    
    return VantageAPI.from_pretrained(model_path)


# Example usage
if __name__ == "__main__":
    # Example demonstrating API usage
    
    schema = """
    CREATE TABLE employees (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        department TEXT,
        salary REAL,
        hire_date DATE
    );
    
    CREATE TABLE departments (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        budget REAL
    );
    """
    
    questions = [
        "How many employees are there?",
        "What is the average salary?",
        "List all departments with their budgets",
        "Who are the highest paid employees in each department?",
    ]
    
    # Create API
    api = VantageAPI.from_pretrained("./checkpoints/medium/best_model")
    
    # Generate SQL for each question
    for question in questions:
        print(f"\nQuestion: {question}")
        sql = api.generate(question, schema)
        print(f"SQL: {sql}")
