"""
Unit tests for data loading and preprocessing
"""

import pytest
from pathlib import Path

from src.data.dataset_loader import DatasetLoader, Text2SQLExample
from src.data.preprocessing import SQLPreprocessor, Tokenizer
from src.data.schema_utils import SchemaParser, SchemaGraph, Table, Column
from src.data.augmentation import DataAugmentor


class TestDatasetLoader:
    """Test dataset loading functionality"""
    
    def test_text2sql_example_creation(self):
        """Test creating Text2SQLExample"""
        example = Text2SQLExample(
            question="How many users?",
            sql="SELECT COUNT(*) FROM users",
            db_id="test_db",
            schema="CREATE TABLE users (id INT, name TEXT);",
            tables=["users"],
            columns={"users": ["id", "name"]},
            foreign_keys=[],
            source="test",
        )
        
        assert example.question == "How many users?"
        assert example.sql == "SELECT COUNT(*) FROM users"
        assert example.tables == ["users"]
    
    def test_dataset_loader_init(self):
        """Test DatasetLoader initialization"""
        loader = DatasetLoader()
        
        assert loader.cache_dir is None
        assert loader.data_dir is None
    
    def test_wikisql_to_sql(self):
        """Test WikiSQL format conversion"""
        loader = DatasetLoader()
        
        sql_dict = {
            "sel": 1,  # Column index
            "agg": 0,  # No aggregation
            "conds": {
                "conditions": [
                    [0, 0, "value"]  # Column 0, op=0 (=), value
                ]
            }
        }
        
        table = {
            "name": "test_table",
            "header": ["id", "name", "email"]
        }
        
        sql = loader._wikisql_to_sql(sql_dict, table)
        
        assert "SELECT name" in sql
        assert "FROM test_table" in sql


class TestSQLPreprocessor:
    """Test SQL preprocessing"""
    
    def test_sql_normalization(self):
        """Test SQL normalization"""
        preprocessor = SQLPreprocessor()
        
        sql = "  select   name  from users where  id=1  "
        normalized = preprocessor.normalize(sql)
        
        # Check keyword uppercase
        assert "SELECT" in normalized
        assert "FROM" in normalized
        assert "WHERE" in normalized
        
        # Check whitespace cleaned
        assert "  " not in normalized
    
    def test_extract_tables(self):
        """Test table extraction"""
        preprocessor = SQLPreprocessor()
        
        sql = "SELECT u.name, o.total FROM users u JOIN orders o ON u.id = o.user_id"
        tables = preprocessor.extract_tables(sql)
        
        assert "users" in tables
        assert "orders" in tables
    
    def test_extract_columns(self):
        """Test column extraction"""
        preprocessor = SQLPreprocessor()
        
        sql = "SELECT name, email, COUNT(*) FROM users"
        columns = preprocessor.extract_columns(sql)
        
        assert "name" in columns
        assert "email" in columns


class TestTokenizer:
    """Test tokenization"""
    
    def test_tokenizer_init(self):
        """Test tokenizer initialization"""
        tokenizer = Tokenizer()
        
        assert tokenizer.vocab_size > 0
        assert tokenizer.pad_token_id is not None
    
    def test_encode_example(self):
        """Test example encoding"""
        tokenizer = Tokenizer()
        
        question = "How many users?"
        schema = "Table users: id, name"
        sql = "SELECT COUNT(*) FROM users"
        
        encoded = tokenizer.encode_example(
            question=question,
            schema=schema,
            sql=sql,
            max_length=512,
        )
        
        assert "input_ids" in encoded
        assert "labels" in encoded
        
        # Check input_ids is list
        assert isinstance(encoded["input_ids"], list)
        assert len(encoded["input_ids"]) > 0
    
    def test_encode_without_sql(self):
        """Test encoding for inference (no target SQL)"""
        tokenizer = Tokenizer()
        
        encoded = tokenizer.encode_example(
            question="Count users",
            schema="Table users: id",
            sql=None,
        )
        
        assert "input_ids" in encoded
        assert "labels" not in encoded
    
    def test_special_tokens(self):
        """Test special tokens are added"""
        tokenizer = Tokenizer()
        
        # Check special tokens in vocab
        special_tokens = ["<schema>", "<query>", "<sql>", "<pad>"]
        
        for token in special_tokens:
            token_id = tokenizer.tokenizer.convert_tokens_to_ids(token)
            assert token_id != tokenizer.tokenizer.unk_token_id


class TestSchemaParser:
    """Test schema parsing"""
    
    def test_parse_create_table(self):
        """Test parsing CREATE TABLE statement"""
        parser = SchemaParser()
        
        create_stmt = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE
        );
        """
        
        table = parser.parse_create_table(create_stmt)
        
        assert table.name == "users"
        assert len(table.columns) == 3
        assert table.columns[0].name == "id"
        assert table.columns[0].is_primary
    
    def test_parse_foreign_keys(self):
        """Test parsing foreign keys"""
        parser = SchemaParser()
        
        create_stmt = """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """
        
        table = parser.parse_create_table(create_stmt)
        
        assert len(table.foreign_keys) == 1
        assert table.foreign_keys[0] == ("user_id", "users", "id")
    
    def test_parse_schema_text(self):
        """Test parsing text-based schema"""
        parser = SchemaParser()
        
        schema_text = "Table users: id (int), name (text) | Table orders: id (int), total (real)"
        tables = parser.parse_schema_text(schema_text)
        
        assert "users" in tables
        assert "orders" in tables
        assert len(tables["users"].columns) == 2


class TestSchemaGraph:
    """Test schema graph construction"""
    
    def test_graph_construction(self):
        """Test building schema graph"""
        tables = {
            "users": Table(
                name="users",
                columns=[Column(name="id", table="users", data_type="INT")],
                primary_keys=["id"],
                foreign_keys=[],
            ),
            "orders": Table(
                name="orders",
                columns=[Column(name="user_id", table="orders", data_type="INT")],
                primary_keys=[],
                foreign_keys=[("user_id", "users", "id")],
            ),
        }
        
        graph = SchemaGraph(tables)
        
        # Check adjacency
        assert "users" in graph.adjacency
        assert "orders" in graph.adjacency["users"]
    
    def test_get_related_tables(self):
        """Test finding related tables"""
        tables = {
            "users": Table("users", [], [], []),
            "orders": Table("orders", [], [], [("user_id", "users", "id")]),
            "items": Table("items", [], [], [("order_id", "orders", "id")]),
        }
        
        graph = SchemaGraph(tables)
        
        related = graph.get_related_tables("users", max_depth=2)
        
        assert "users" in related
        assert "orders" in related
        assert "items" in related  # 2 hops away


class TestDataAugmentor:
    """Test data augmentation"""
    
    def test_augmentor_init(self):
        """Test augmentor initialization"""
        augmentor = DataAugmentor(
            paraphrase_prob=0.3,
            schema_perturb_prob=0.1,
        )
        
        assert augmentor.paraphrase_prob == 0.3
        assert augmentor.schema_perturb_prob == 0.1
    
    def test_paraphrase_question(self):
        """Test question paraphrasing"""
        augmentor = DataAugmentor(seed=42)
        
        question = "What are the names of all users?"
        paraphrased = augmentor._paraphrase_question(question)
        
        # Should be different (most of the time)
        # Or at least processed
        assert isinstance(paraphrased, str)
        assert len(paraphrased) > 0
    
    def test_augment_example(self):
        """Test full example augmentation"""
        augmentor = DataAugmentor(
            paraphrase_prob=1.0,  # Always paraphrase for test
            schema_perturb_prob=0.0,
            seed=42,
        )
        
        example = Text2SQLExample(
            question="List all users",
            sql="SELECT * FROM users",
            db_id="test",
            schema="Table users: id, name",
            tables=["users"],
            columns={"users": ["id", "name"]},
            foreign_keys=[],
            source="test",
        )
        
        augmented = augmentor.augment(example)
        
        # Should return valid example
        assert isinstance(augmented, Text2SQLExample)
        assert augmented.db_id == "test"
    
    def test_transform_sql(self):
        """Test SQL transformation"""
        augmentor = DataAugmentor(seed=42)
        
        sql = "SELECT name FROM users WHERE id = 1 AND active = 1"
        transformed = augmentor._transform_sql(sql)
        
        # Should return valid SQL
        assert isinstance(transformed, str)
        assert "SELECT" in transformed.upper()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
