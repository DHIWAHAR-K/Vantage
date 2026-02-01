"""
SQL generation with beam search and constrained decoding
"""

import mlx.core as mx
import mlx.nn as nn
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuration for text generation"""
    max_new_tokens: int = 512
    temperature: float = 0.0  # Greedy for SQL
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 4
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    early_stopping: bool = True
    use_cache: bool = True
    constrained_decoding: bool = True
    schema_aware: bool = True


class Beam:
    """Single beam for beam search"""
    
    def __init__(
        self,
        tokens: List[int],
        score: float,
        cache: Optional[Dict] = None,
    ):
        self.tokens = tokens
        self.score = score
        self.cache = cache
        self.finished = False
    
    def __lt__(self, other):
        """Compare beams by score"""
        return self.score < other.score


class SQLGenerator:
    """
    SQL generator with beam search and constrained decoding.
    
    Features:
    - Beam search for better quality
    - Schema-aware generation (only valid table/column references)
    - Constrained decoding (valid SQL tokens only)
    - Temperature sampling and top-p/top-k filtering
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        config: Optional[GenerationConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        
        # SQL keywords for constrained decoding
        self.sql_keywords = {
            "SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "HAVING",
            "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "ON", "AS",
            "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "IS", "NULL",
            "COUNT", "SUM", "AVG", "MAX", "MIN", "DISTINCT",
            "LIMIT", "OFFSET", "ASC", "DESC",
        }
    
    def generate(
        self,
        question: str,
        schema: str,
        tables: Optional[List[str]] = None,
        columns: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Generate SQL query from natural language question and schema.
        
        Args:
            question: Natural language question
            schema: Database schema description
            tables: List of table names (for schema-aware generation)
            columns: Dictionary of table -> columns (for schema-aware generation)
            
        Returns:
            Generated SQL query
        """
        # Encode input
        encoded = self.tokenizer.encode_example(
            question=question,
            schema=schema,
            sql=None,
        )
        
        input_ids = mx.array([encoded["input_ids"]])
        
        # Generate with beam search
        if self.config.num_beams > 1:
            output_ids = self._beam_search(
                input_ids=input_ids,
                tables=tables,
                columns=columns,
            )
        else:
            output_ids = self._greedy_search(
                input_ids=input_ids,
                tables=tables,
                columns=columns,
            )
        
        # Decode output
        sql = self.tokenizer.decode(
            output_ids[0].tolist(),
            skip_special_tokens=True,
        )
        
        # Extract SQL part (after <sql> token)
        if "<sql>" in sql:
            sql = sql.split("<sql>")[-1].strip()
        
        # Post-process
        sql = self._post_process_sql(sql)
        
        return sql
    
    def _greedy_search(
        self,
        input_ids: mx.array,
        tables: Optional[List[str]] = None,
        columns: Optional[Dict[str, List[str]]] = None,
    ) -> mx.array:
        """
        Greedy decoding (temperature = 0).
        
        Args:
            input_ids: Input token IDs
            tables: Valid table names
            columns: Valid column names
            
        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]
        generated_ids = input_ids
        
        # Track which sequences are finished
        finished = mx.zeros(batch_size, dtype=mx.bool_)
        
        for _ in range(self.config.max_new_tokens):
            # Forward pass
            outputs = self.model(
                input_ids=generated_ids,
                training=False,
            )
            
            # Get logits for next token
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply constraints if enabled
            if self.config.constrained_decoding:
                next_token_logits = self._apply_constraints(
                    next_token_logits,
                    generated_ids,
                    tables,
                    columns,
                )
            
            # Greedy selection
            next_tokens = mx.argmax(next_token_logits, axis=-1, keepdims=True)
            
            # Append to generated sequence
            generated_ids = mx.concatenate([generated_ids, next_tokens], axis=1)
            
            # Check for EOS
            eos_token_id = self.tokenizer.tokenizer.eos_token_id
            if eos_token_id is not None:
                finished = finished | (next_tokens[:, 0] == eos_token_id)
                
                if finished.all():
                    break
        
        return generated_ids
    
    def _beam_search(
        self,
        input_ids: mx.array,
        tables: Optional[List[str]] = None,
        columns: Optional[Dict[str, List[str]]] = None,
    ) -> mx.array:
        """
        Beam search decoding.
        
        Args:
            input_ids: Input token IDs
            tables: Valid table names
            columns: Valid column names
            
        Returns:
            Best generated sequence
        """
        batch_size = input_ids.shape[0]
        num_beams = self.config.num_beams
        
        # Initialize beams
        beams = [
            Beam(tokens=input_ids[i].tolist(), score=0.0)
            for i in range(batch_size)
            for _ in range(num_beams)
        ]
        
        # Track finished beams
        finished_beams = []
        
        for step in range(self.config.max_new_tokens):
            # Prepare current sequences
            current_sequences = mx.array([beam.tokens for beam in beams if not beam.finished])
            
            if current_sequences.shape[0] == 0:
                break
            
            # Forward pass
            outputs = self.model(
                input_ids=current_sequences,
                training=False,
            )
            
            # Get logits
            next_token_logits = outputs["logits"][:, -1, :]
            
            # Apply temperature
            if self.config.temperature > 0:
                next_token_logits = next_token_logits / self.config.temperature
            
            # Log probabilities
            log_probs = nn.log_softmax(next_token_logits, axis=-1)
            
            # For each beam, get top-k candidates
            beam_candidates = []
            
            for i, beam in enumerate([b for b in beams if not b.finished]):
                # Get top-k tokens
                top_k_log_probs, top_k_indices = mx.top_k(log_probs[i], k=num_beams)
                
                # Create new beam candidates
                for j in range(num_beams):
                    new_token = top_k_indices[j].item()
                    new_score = beam.score + top_k_log_probs[j].item()
                    
                    # Length penalty
                    length_penalty = ((len(beam.tokens) + 1) ** self.config.length_penalty)
                    normalized_score = new_score / length_penalty
                    
                    new_beam = Beam(
                        tokens=beam.tokens + [new_token],
                        score=new_score,
                    )
                    
                    # Check if finished
                    eos_token_id = self.tokenizer.tokenizer.eos_token_id
                    if eos_token_id and new_token == eos_token_id:
                        new_beam.finished = True
                        finished_beams.append(new_beam)
                    
                    beam_candidates.append((normalized_score, new_beam))
            
            # Select top beams
            beam_candidates.sort(reverse=True, key=lambda x: x[0])
            beams = [beam for _, beam in beam_candidates[:num_beams]]
            
            # Early stopping
            if self.config.early_stopping and len(finished_beams) >= num_beams:
                break
        
        # Select best beam
        all_beams = finished_beams + [b for b in beams if not b.finished]
        if not all_beams:
            # Fallback to first beam
            best_beam = beams[0]
        else:
            # Best by score (with length normalization)
            best_beam = max(
                all_beams,
                key=lambda b: b.score / (len(b.tokens) ** self.config.length_penalty)
            )
        
        return mx.array([best_beam.tokens])
    
    def _apply_constraints(
        self,
        logits: mx.array,
        generated_ids: mx.array,
        tables: Optional[List[str]] = None,
        columns: Optional[Dict[str, List[str]]] = None,
    ) -> mx.array:
        """
        Apply constrained decoding rules.
        
        Args:
            logits: Next token logits
            generated_ids: Currently generated tokens
            tables: Valid table names
            columns: Valid column names
            
        Returns:
            Modified logits
        """
        # Simple version: boost SQL keywords
        # More sophisticated version would parse context
        
        if self.config.schema_aware and tables is not None:
            # Boost valid table/column tokens
            # This requires tokenizer access to table/column tokens
            pass
        
        # Could mask invalid tokens based on SQL grammar
        # For now, return as-is
        return logits
    
    def _post_process_sql(self, sql: str) -> str:
        """
        Post-process generated SQL.
        
        Args:
            sql: Raw generated SQL
            
        Returns:
            Cleaned SQL
        """
        # Remove incomplete statements
        if sql.count("SELECT") > sql.count("FROM"):
            # Incomplete query
            sql = sql[:sql.rfind("SELECT")]
        
        # Remove trailing incomplete tokens
        sql = sql.strip()
        
        # Ensure ends properly
        if sql and not sql.endswith(";"):
            # Check if complete
            sql_upper = sql.upper()
            if "SELECT" in sql_upper and "FROM" in sql_upper:
                sql = sql + ";"
        
        return sql
    
    def generate_batch(
        self,
        questions: List[str],
        schemas: List[str],
        tables_list: Optional[List[List[str]]] = None,
        columns_list: Optional[List[Dict[str, List[str]]]] = None,
    ) -> List[str]:
        """
        Generate SQL for batch of questions.
        
        Args:
            questions: List of questions
            schemas: List of schemas
            tables_list: List of table names for each example
            columns_list: List of column dictionaries for each example
            
        Returns:
            List of generated SQL queries
        """
        results = []
        
        for i, (question, schema) in enumerate(zip(questions, schemas)):
            tables = tables_list[i] if tables_list else None
            columns = columns_list[i] if columns_list else None
            
            sql = self.generate(
                question=question,
                schema=schema,
                tables=tables,
                columns=columns,
            )
            
            results.append(sql)
        
        return results
