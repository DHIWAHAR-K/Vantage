"""
Benchmark evaluation for text-to-SQL models.

Evaluate on standard benchmarks:
- Spider dev/test
- WikiSQL test
- SynSQL holdout
"""

import sys
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class EvaluationResult:
    """Results from benchmark evaluation."""
    exact_match: float
    execution_accuracy: float
    total_examples: int
    correct_em: int
    correct_ex: int


class SQLEvaluator:
    """
    Evaluate text-to-SQL predictions.
    
    Metrics:
    - Exact Match (EM): Predicted SQL exactly matches ground truth
    - Execution Accuracy (EX): Predicted SQL produces same result as ground truth
    """
    
    def __init__(self):
        pass
    
    def normalize_sql(self, sql: str) -> str:
        """
        Normalize SQL for comparison.
        
        Args:
            sql: SQL query string
            
        Returns:
            Normalized SQL (lowercase, stripped, standardized spacing)
        """
        # Remove extra whitespace
        sql = ' '.join(sql.split())
        # Lowercase
        sql = sql.lower()
        # Remove trailing semicolon
        sql = sql.rstrip(';')
        return sql
    
    def exact_match(self, predicted: str, ground_truth: str) -> bool:
        """
        Check if predicted SQL exactly matches ground truth.
        
        Args:
            predicted: Predicted SQL
            ground_truth: Ground truth SQL
            
        Returns:
            True if exact match after normalization
        """
        pred_norm = self.normalize_sql(predicted)
        gt_norm = self.normalize_sql(ground_truth)
        return pred_norm == gt_norm
    
    def execution_match(
        self,
        predicted: str,
        ground_truth: str,
        db_path: str
    ) -> bool:
        """
        Check if predicted SQL produces same result as ground truth.
        
        Args:
            predicted: Predicted SQL
            ground_truth: Ground truth SQL
            db_path: Path to SQLite database
            
        Returns:
            True if execution results match
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Execute both queries
            cursor.execute(predicted)
            pred_result = cursor.fetchall()
            
            cursor.execute(ground_truth)
            gt_result = cursor.fetchall()
            
            conn.close()
            
            # Compare results
            return pred_result == gt_result
            
        except Exception as e:
            # If either query fails, not a match
            return False
    
    def evaluate_predictions(
        self,
        predictions: List[Tuple[str, str, str]],
        use_execution: bool = True
    ) -> EvaluationResult:
        """
        Evaluate a list of predictions.
        
        Args:
            predictions: List of (predicted_sql, ground_truth_sql, db_path)
            use_execution: Whether to compute execution accuracy
            
        Returns:
            EvaluationResult with metrics
        """
        total = len(predictions)
        correct_em = 0
        correct_ex = 0
        
        for pred_sql, gt_sql, db_path in predictions:
            # Exact match
            if self.exact_match(pred_sql, gt_sql):
                correct_em += 1
            
            # Execution match
            if use_execution and db_path:
                if self.execution_match(pred_sql, gt_sql, db_path):
                    correct_ex += 1
        
        return EvaluationResult(
            exact_match=correct_em / total if total > 0 else 0.0,
            execution_accuracy=correct_ex / total if total > 0 else 0.0,
            total_examples=total,
            correct_em=correct_em,
            correct_ex=correct_ex
        )


def load_spider_dev(data_dir: str = "data/spider") -> List[Dict]:
    """
    Load Spider dev set.
    
    Args:
        data_dir: Path to Spider dataset
        
    Returns:
        List of examples
    """
    dev_file = Path(data_dir) / "dev.json"
    
    if not dev_file.exists():
        print(f"WARNING: Spider dev not found at {dev_file}")
        return []
    
    with open(dev_file, 'r') as f:
        data = json.load(f)
    
    return data


def load_wikisql_test(data_dir: str = "data/wikisql") -> List[Dict]:
    """
    Load WikiSQL test set.
    
    Args:
        data_dir: Path to WikiSQL dataset
        
    Returns:
        List of examples
    """
    test_file = Path(data_dir) / "test.jsonl"
    
    if not test_file.exists():
        print(f"WARNING: WikiSQL test not found at {test_file}")
        return []
    
    examples = []
    with open(test_file, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    return examples


def evaluate_model(
    model,
    tokenizer,
    benchmark: str = "spider",
    data_dir: str = "data"
):
    """
    Evaluate model on a benchmark.
    
    Args:
        model: Fine-tuned T5 model
        tokenizer: T5 tokenizer
        benchmark: "spider", "wikisql", or "synsql"
        data_dir: Path to data directory
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING ON {benchmark.upper()}")
    print(f"{'='*60}\n")
    
    # Load benchmark data
    if benchmark == "spider":
        examples = load_spider_dev(f"{data_dir}/spider")
    elif benchmark == "wikisql":
        examples = load_wikisql_test(f"{data_dir}/wikisql")
    else:
        print(f"Unknown benchmark: {benchmark}")
        return
    
    if not examples:
        print("No examples found!")
        return
    
    print(f"Loaded {len(examples)} examples")
    
    # Generate predictions
    print("Generating predictions...")
    predictions = []
    
    for i, example in enumerate(examples):
        if i % 100 == 0:
            print(f"Progress: {i}/{len(examples)}")
        
        # Extract fields (format depends on benchmark)
        question = example.get("question", "")
        ground_truth = example.get("query", "")
        db_id = example.get("db_id", "")
        
        # Generate prediction
        # TODO: Implement actual inference
        predicted_sql = "SELECT * FROM table"  # Placeholder
        
        # Get database path
        db_path = f"{data_dir}/{benchmark}/database/{db_id}/{db_id}.sqlite"
        
        predictions.append((predicted_sql, ground_truth, db_path))
    
    # Evaluate
    print("\nComputing metrics...")
    evaluator = SQLEvaluator()
    results = evaluator.evaluate_predictions(predictions)
    
    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total examples: {results.total_examples}")
    print(f"Exact Match: {results.exact_match*100:.2f}% ({results.correct_em}/{results.total_examples})")
    print(f"Execution Accuracy: {results.execution_accuracy*100:.2f}% ({results.correct_ex}/{results.total_examples})")
    print(f"{'='*60}\n")
    
    return results


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate T5 text-to-SQL model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["spider", "wikisql", "all"],
        default="all",
        help="Which benchmark to evaluate on"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    # Load model
    print("Loading model from checkpoint...")
    # TODO: Implement model loading
    print(f"Checkpoint: {args.checkpoint}")
    
    # Run evaluation
    if args.benchmark == "all":
        benchmarks = ["spider", "wikisql"]
    else:
        benchmarks = [args.benchmark]
    
    for benchmark in benchmarks:
        evaluate_model(None, None, benchmark, args.data_dir)


if __name__ == "__main__":
    main()
