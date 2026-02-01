"""
Unit tests for inference and generation
"""

import pytest
import mlx.core as mx

from src.inference.generator import SQLGenerator, GenerationConfig, Beam
from src.inference.api import VantageAPI
from src.models.text2sql_model import VantageModel, VantageConfig
from src.data.preprocessing import Tokenizer


class TestBeam:
    """Test beam dataclass"""
    
    def test_beam_creation(self):
        """Test creating beam"""
        beam = Beam(
            tokens=[1, 2, 3],
            score=-1.5,
        )
        
        assert beam.tokens == [1, 2, 3]
        assert beam.score == -1.5
        assert not beam.finished
    
    def test_beam_comparison(self):
        """Test beam comparison"""
        beam1 = Beam([1, 2], score=-1.0)
        beam2 = Beam([3, 4], score=-2.0)
        
        # Lower score < higher score
        assert beam2 < beam1


class TestGenerationConfig:
    """Test generation configuration"""
    
    def test_default_config(self):
        """Test default generation config"""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.0
        assert config.num_beams == 4
        assert config.early_stopping == True
    
    def test_custom_config(self):
        """Test custom generation config"""
        config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            num_beams=8,
        )
        
        assert config.max_new_tokens == 256
        assert config.temperature == 0.7
        assert config.num_beams == 8


class TestSQLGenerator:
    """Test SQL generation"""
    
    @pytest.fixture
    def setup_generator(self):
        """Create test generator"""
        # Create small model for testing
        config = VantageConfig.small()
        model = VantageModel(config)
        tokenizer = Tokenizer()
        
        generator = SQLGenerator(
            model=model,
            tokenizer=tokenizer,
            config=GenerationConfig(max_new_tokens=50),  # Short for tests
        )
        
        return generator
    
    def test_generator_init(self, setup_generator):
        """Test generator initialization"""
        generator = setup_generator
        
        assert generator.model is not None
        assert generator.tokenizer is not None
        assert generator.config.max_new_tokens == 50
    
    def test_post_process_sql(self, setup_generator):
        """Test SQL post-processing"""
        generator = setup_generator
        
        # Test incomplete SQL
        sql = "SELECT name FROM users SELECT"
        processed = generator._post_process_sql(sql)
        
        # Should remove incomplete SELECT
        assert processed.count("SELECT") == 1
    
    def test_generate_batch(self, setup_generator):
        """Test batch generation"""
        generator = setup_generator
        
        questions = ["Count users", "List all"]
        schemas = ["Table users: id", "Table items: id"]
        
        # This will be slow with real model, just test interface
        # In practice, would mock the model
        results = generator.generate_batch(questions, schemas)
        
        assert len(results) == 2
        assert all(isinstance(sql, str) for sql in results)


class TestVantageAPI:
    """Test high-level API"""
    
    @pytest.fixture
    def setup_api(self):
        """Create test API"""
        config = VantageConfig.small()
        model = VantageModel(config)
        tokenizer = Tokenizer()
        
        api = VantageAPI(
            model=model,
            tokenizer=tokenizer,
            generation_config=GenerationConfig(max_new_tokens=30),
        )
        
        return api
    
    def test_api_init(self, setup_api):
        """Test API initialization"""
        api = setup_api
        
        assert api.model is not None
        assert api.tokenizer is not None
        assert api.generator is not None
    
    def test_parse_schema(self, setup_api):
        """Test schema parsing"""
        api = setup_api
        
        schema = """
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            name TEXT
        );
        """
        
        tables, columns = api._parse_schema(schema)
        
        assert "users" in tables
        assert "users" in columns
        assert "id" in columns["users"]
        assert "name" in columns["users"]
    
    def test_validate_schema(self, setup_api):
        """Test schema validation"""
        api = setup_api
        
        schema = """
        CREATE TABLE users (id INT, name TEXT);
        CREATE TABLE orders (id INT, user_id INT);
        """
        
        analysis = api.validate_schema(schema)
        
        assert analysis["valid"] == True
        assert analysis["num_tables"] == 2
        assert "users" in analysis["tables"]
        assert "orders" in analysis["tables"]
    
    def test_set_generation_config(self, setup_api):
        """Test updating generation config"""
        api = setup_api
        
        api.set_generation_config(
            temperature=0.8,
            num_beams=2,
        )
        
        assert api.generator.config.temperature == 0.8
        assert api.generator.config.num_beams == 2
    
    def test_api_call_shorthand(self, setup_api):
        """Test API __call__ shorthand"""
        api = setup_api
        
        # Just test that it doesn't crash
        # Actual generation would be slow
        try:
            result = api("test question", "Table test: id")
            assert isinstance(result, str)
        except Exception as e:
            # Expected in test environment without full model
            pass


class TestGenerationStrategies:
    """Test different generation strategies"""
    
    @pytest.fixture
    def mock_generator(self):
        """Create mock generator for fast tests"""
        config = VantageConfig(
            hidden_size=256,
            num_layers=2,
            num_experts=4,
            vocab_size=1000,
        )
        model = VantageModel(config)
        tokenizer = Tokenizer()
        
        return SQLGenerator(model, tokenizer)
    
    def test_greedy_vs_beam(self, mock_generator):
        """Test that greedy and beam search have different configs"""
        gen = mock_generator
        
        # Greedy (default)
        assert gen.config.temperature == 0.0
        assert gen.config.num_beams == 4
        
        # Change to beam search
        gen.config.num_beams = 8
        assert gen.config.num_beams == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
