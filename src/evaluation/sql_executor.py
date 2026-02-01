"""
Safe SQL execution for evaluation
Sandbox environment with timeout protection
"""

import sqlite3
import signal
from pathlib import Path
from typing import List, Any, Optional
import tempfile
import shutil


class TimeoutError(Exception):
    """Raised when query execution times out"""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout"""
    raise TimeoutError("Query execution timed out")


class SQLExecutor:
    """
    Safe SQL executor for evaluation.
    
    Features:
    - Read-only execution
    - Timeout protection
    - Sandboxed environment
    - Result caching
    """
    
    def __init__(
        self,
        db_dir: str,
        timeout: float = 5.0,
        max_result_size: int = 1000,
    ):
        """
        Initialize SQL executor.
        
        Args:
            db_dir: Directory containing database files
            timeout: Maximum execution time in seconds
            max_result_size: Maximum number of rows to return
        """
        self.db_dir = Path(db_dir)
        self.timeout = timeout
        self.max_result_size = max_result_size
        
        # Cache for database connections
        self.connections = {}
    
    def execute(
        self,
        sql: str,
        db_id: str,
        return_columns: bool = False,
    ) -> List[Any]:
        """
        Execute SQL query on specified database.
        
        Args:
            sql: SQL query to execute
            db_id: Database identifier
            return_columns: Whether to return column names
            
        Returns:
            Query results as list of rows
        """
        # Get database path
        db_path = self.db_dir / f"{db_id}.sqlite"
        
        if not db_path.exists():
            db_path = self.db_dir / f"{db_id}.db"
        
        if not db_path.exists():
            raise ValueError(f"Database not found: {db_id}")
        
        # Get or create connection
        conn = self._get_connection(str(db_path))
        
        try:
            # Set timeout
            if hasattr(signal, 'SIGALRM'):
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(self.timeout))
            
            # Execute query
            cursor = conn.cursor()
            cursor.execute(sql)
            
            # Fetch results (limited)
            results = cursor.fetchmany(self.max_result_size)
            
            # Cancel timeout
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
            
            # Return with or without column names
            if return_columns:
                columns = [desc[0] for desc in cursor.description] if cursor.description else []
                return columns, results
            else:
                return results
        
        except TimeoutError:
            raise TimeoutError(f"Query exceeded timeout of {self.timeout}s")
        
        except sqlite3.Error as e:
            raise RuntimeError(f"SQL execution error: {e}")
        
        finally:
            # Cancel timeout if still set
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)
    
    def _get_connection(self, db_path: str) -> sqlite3.Connection:
        """
        Get database connection (cached).
        
        Args:
            db_path: Path to database file
            
        Returns:
            SQLite connection
        """
        if db_path not in self.connections:
            # Create read-only connection
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            
            # Set pragmas for safety
            conn.execute("PRAGMA query_only = ON")
            
            self.connections[db_path] = conn
        
        return self.connections[db_path]
    
    def execute_batch(
        self,
        sql_list: List[str],
        db_ids: List[str],
    ) -> List[Optional[List[Any]]]:
        """
        Execute batch of SQL queries.
        
        Args:
            sql_list: List of SQL queries
            db_ids: List of database identifiers
            
        Returns:
            List of results (None for failed queries)
        """
        results = []
        
        for sql, db_id in zip(sql_list, db_ids):
            try:
                result = self.execute(sql, db_id)
                results.append(result)
            except Exception as e:
                results.append(None)
        
        return results
    
    def validate_sql(self, sql: str, db_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate SQL without executing.
        
        Args:
            sql: SQL query
            db_id: Database identifier
            
        Returns:
            (is_valid, error_message)
        """
        try:
            db_path = self.db_dir / f"{db_id}.sqlite"
            if not db_path.exists():
                db_path = self.db_dir / f"{db_id}.db"
            
            conn = self._get_connection(str(db_path))
            
            # Use EXPLAIN to validate without executing
            cursor = conn.cursor()
            cursor.execute(f"EXPLAIN {sql}")
            
            return True, None
        
        except Exception as e:
            return False, str(e)
    
    def close(self):
        """Close all database connections"""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()


class SandboxExecutor(SQLExecutor):
    """
    SQL executor with additional sandboxing.
    
    Creates temporary copy of database for execution to prevent any
    potential modifications.
    """
    
    def __init__(
        self,
        db_dir: str,
        timeout: float = 5.0,
        max_result_size: int = 1000,
    ):
        super().__init__(db_dir, timeout, max_result_size)
        
        # Create temporary directory for sandbox
        self.sandbox_dir = Path(tempfile.mkdtemp(prefix="vantage_sandbox_"))
    
    def execute(
        self,
        sql: str,
        db_id: str,
        return_columns: bool = False,
    ) -> List[Any]:
        """
        Execute SQL in sandboxed environment.
        
        Args:
            sql: SQL query
            db_id: Database identifier
            return_columns: Whether to return column names
            
        Returns:
            Query results
        """
        # Copy database to sandbox
        src_path = self.db_dir / f"{db_id}.sqlite"
        if not src_path.exists():
            src_path = self.db_dir / f"{db_id}.db"
        
        sandbox_path = self.sandbox_dir / f"{db_id}.db"
        
        if not sandbox_path.exists():
            shutil.copy2(src_path, sandbox_path)
        
        # Execute on sandbox copy
        # Temporarily override db_dir
        original_db_dir = self.db_dir
        self.db_dir = self.sandbox_dir
        
        try:
            result = super().execute(sql, db_id, return_columns)
            return result
        finally:
            self.db_dir = original_db_dir
    
    def cleanup_sandbox(self):
        """Remove sandbox directory"""
        if self.sandbox_dir.exists():
            shutil.rmtree(self.sandbox_dir)
    
    def __del__(self):
        """Cleanup sandbox on deletion"""
        super().__del__()
        self.cleanup_sandbox()
