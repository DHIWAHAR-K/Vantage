"""
Pytest configuration and fixtures
"""

import pytest
import mlx.core as mx


@pytest.fixture(scope="session")
def mlx_seed():
    """Set random seed for reproducibility"""
    mx.random.seed(42)
    yield
    
@pytest.fixture
def sample_schema():
    """Sample database schema for testing"""
    return """
    CREATE TABLE users (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        email TEXT UNIQUE,
        created_at TIMESTAMP
    );
    
    CREATE TABLE orders (
        id INTEGER PRIMARY KEY,
        user_id INTEGER,
        total REAL,
        status TEXT,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );
    """

@pytest.fixture
def sample_questions():
    """Sample questions for testing"""
    return [
        "How many users are there?",
        "What is the total revenue?",
        "List all active orders",
        "Who are the top 10 customers by order value?",
    ]

@pytest.fixture
def sample_sqls():
    """Sample SQL queries for testing"""
    return [
        "SELECT COUNT(*) FROM users",
        "SELECT SUM(total) FROM orders",
        "SELECT * FROM orders WHERE status = 'active'",
        "SELECT u.name, SUM(o.total) as total FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name ORDER BY total DESC LIMIT 10",
    ]
