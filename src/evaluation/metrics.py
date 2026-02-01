"""
Evaluation metrics for text-to-SQL
Includes exact match, execution accuracy, and component-wise metrics
"""

import sqlparse
from typing import List, Dict, Tuple, Set
import re
from collections import defaultdict


class ExactMatch:
    """
    Exact match metric after SQL normalization.
    
    Compares predicted and gold SQL after:
    - Keyword normalization
    - Whitespace normalization
    - Alias removal
    """
    
    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
    
    def normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for comparison.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL
        """
        try:
            # Parse and format
            formatted = sqlparse.format(
                sql,
                keyword_case="upper",
                identifier_case="lower" if not self.case_sensitive else None,
                strip_comments=True,
                reindent=False,
            )
        except:
            formatted = sql
        
        # Remove extra whitespace
        formatted = re.sub(r'\s+', ' ', formatted).strip()
        
        if not self.case_sensitive:
            formatted = formatted.lower()
        
        return formatted
    
    def compute(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute exact match accuracy.
        
        Args:
            predictions: List of predicted SQL queries
            references: List of gold SQL queries
            
        Returns:
            Dictionary with exact match score
        """
        assert len(predictions) == len(references), "Must have same number of predictions and references"
        
        correct = 0
        total = len(predictions)
        
        for pred, ref in zip(predictions, references):
            norm_pred = self.normalize_sql(pred)
            norm_ref = self.normalize_sql(ref)
            
            if norm_pred == norm_ref:
                correct += 1
        
        return {
            "exact_match": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
        }


class ExecutionAccuracy:
    """
    Execution accuracy metric.
    
    Executes predicted and gold SQL on database and compares results.
    More lenient than exact match - considers queries equivalent if they
    produce the same results.
    """
    
    def __init__(self, sql_executor):
        self.sql_executor = sql_executor
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
        db_ids: List[str],
    ) -> Dict[str, float]:
        """
        Compute execution accuracy.
        
        Args:
            predictions: List of predicted SQL queries
            references: List of gold SQL queries
            db_ids: List of database identifiers
            
        Returns:
            Dictionary with execution accuracy
        """
        assert len(predictions) == len(references) == len(db_ids)
        
        correct = 0
        total = 0
        errors = 0
        
        for pred, ref, db_id in zip(predictions, references, db_ids):
            try:
                # Execute both queries
                pred_result = self.sql_executor.execute(pred, db_id)
                ref_result = self.sql_executor.execute(ref, db_id)
                
                # Compare results (order-independent)
                if self._results_match(pred_result, ref_result):
                    correct += 1
                
                total += 1
            except Exception as e:
                # Query execution failed
                errors += 1
                total += 1
        
        return {
            "execution_accuracy": correct / total if total > 0 else 0.0,
            "correct": correct,
            "total": total,
            "errors": errors,
            "error_rate": errors / total if total > 0 else 0.0,
        }
    
    def _results_match(self, result1: List, result2: List) -> bool:
        """Check if two query results match (order-independent)"""
        if len(result1) != len(result2):
            return False
        
        # Convert to sets for comparison (handles order)
        set1 = {tuple(row) if isinstance(row, (list, tuple)) else row for row in result1}
        set2 = {tuple(row) if isinstance(row, (list, tuple)) else row for row in result2}
        
        return set1 == set2


class ValidSQL:
    """
    Check if SQL is syntactically valid.
    """
    
    def compute(self, predictions: List[str]) -> Dict[str, float]:
        """
        Compute percentage of valid SQL queries.
        
        Args:
            predictions: List of predicted SQL queries
            
        Returns:
            Dictionary with valid SQL percentage
        """
        valid = 0
        total = len(predictions)
        
        for sql in predictions:
            try:
                # Try to parse
                parsed = sqlparse.parse(sql)
                if parsed and len(parsed) > 0:
                    valid += 1
            except:
                pass
        
        return {
            "valid_sql": valid / total if total > 0 else 0.0,
            "valid": valid,
            "total": total,
        }


class ComponentMatch:
    """
    Component-wise matching metric.
    
    Evaluates accuracy of individual SQL components:
    - SELECT clause
    - WHERE clause
    - GROUP BY clause
    - ORDER BY clause
    - JOIN operations
    """
    
    SQL_KEYWORDS = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "LIMIT", "JOIN"]
    
    def extract_components(self, sql: str) -> Dict[str, str]:
        """
        Extract SQL components.
        
        Args:
            sql: SQL query
            
        Returns:
            Dictionary of components
        """
        components = defaultdict(str)
        
        # Normalize
        sql = sql.upper()
        sql = re.sub(r'\s+', ' ', sql).strip()
        
        # Extract SELECT
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.DOTALL)
        if select_match:
            components["SELECT"] = select_match.group(1).strip()
        
        # Extract FROM
        from_match = re.search(r'FROM\s+(.+?)(?:\s+WHERE|\s+GROUP|\s+ORDER|\s+LIMIT|$)', sql, re.DOTALL)
        if from_match:
            components["FROM"] = from_match.group(1).strip()
        
        # Extract WHERE
        where_match = re.search(r'WHERE\s+(.+?)(?:\s+GROUP|\s+ORDER|\s+LIMIT|$)', sql, re.DOTALL)
        if where_match:
            components["WHERE"] = where_match.group(1).strip()
        
        # Extract GROUP BY
        group_match = re.search(r'GROUP BY\s+(.+?)(?:\s+HAVING|\s+ORDER|\s+LIMIT|$)', sql, re.DOTALL)
        if group_match:
            components["GROUP BY"] = group_match.group(1).strip()
        
        # Extract ORDER BY
        order_match = re.search(r'ORDER BY\s+(.+?)(?:\s+LIMIT|$)', sql, re.DOTALL)
        if order_match:
            components["ORDER BY"] = order_match.group(1).strip()
        
        # Check for JOINs
        if "JOIN" in sql:
            components["JOIN"] = "1"
        
        return components
    
    def compute(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        Compute component-wise accuracy.
        
        Args:
            predictions: List of predicted SQL queries
            references: List of gold SQL queries
            
        Returns:
            Dictionary with component accuracies
        """
        component_correct = defaultdict(int)
        component_total = defaultdict(int)
        
        for pred, ref in zip(predictions, references):
            pred_components = self.extract_components(pred)
            ref_components = self.extract_components(ref)
            
            # Check each component
            for component in self.SQL_KEYWORDS:
                if component in ref_components:
                    component_total[component] += 1
                    
                    if component in pred_components:
                        # Compare component text (after normalization)
                        pred_text = re.sub(r'\s+', ' ', pred_components[component]).strip()
                        ref_text = re.sub(r'\s+', ' ', ref_components[component]).strip()
                        
                        if pred_text == ref_text:
                            component_correct[component] += 1
        
        # Compute accuracies
        results = {}
        for component in self.SQL_KEYWORDS:
            if component_total[component] > 0:
                accuracy = component_correct[component] / component_total[component]
                results[f"{component.lower().replace(' ', '_')}_accuracy"] = accuracy
        
        return results


def compute_metrics(
    predictions: List[str],
    references: List[str],
    db_ids: Optional[List[str]] = None,
    sql_executor=None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted SQL queries
        references: List of gold SQL queries
        db_ids: List of database identifiers (for execution accuracy)
        sql_executor: SQL executor instance (for execution accuracy)
        
    Returns:
        Dictionary with all metrics
    """
    all_metrics = {}
    
    # Exact match
    em = ExactMatch()
    all_metrics.update(em.compute(predictions, references))
    
    # Valid SQL
    valid_sql = ValidSQL()
    all_metrics.update(valid_sql.compute(predictions))
    
    # Component match
    component_match = ComponentMatch()
    all_metrics.update(component_match.compute(predictions, references))
    
    # Execution accuracy (if available)
    if sql_executor is not None and db_ids is not None:
        exec_acc = ExecutionAccuracy(sql_executor)
        all_metrics.update(exec_acc.compute(predictions, references, db_ids))
    
    return all_metrics


def compute_aggregated_metrics(
    metrics_list: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate metrics across multiple batches or datasets.
    
    Args:
        metrics_list: List of metric dictionaries
        
    Returns:
        Aggregated metrics
    """
    # Average metrics
    aggregated = defaultdict(list)
    
    for metrics in metrics_list:
        for key, value in metrics.items():
            aggregated[key].append(value)
    
    # Compute means
    result = {}
    for key, values in aggregated.items():
        result[key] = sum(values) / len(values)
    
    return result
