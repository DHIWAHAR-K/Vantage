"""
Benchmark evaluation on standard text-to-SQL datasets
"""

from typing import List, Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .metrics import compute_metrics, ExactMatch, ExecutionAccuracy, ValidSQL, ComponentMatch
from .sql_executor import SQLExecutor
from ..data.dataset_loader import DatasetLoader, Text2SQLExample


class Benchmark:
    """
    Benchmark evaluation framework for text-to-SQL models.
    
    Supports:
    - Spider dev/test
    - BIRD-SQL
    - WikiSQL
    - CoSQL
    - Custom datasets
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        sql_executor: Optional[SQLExecutor] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.sql_executor = sql_executor
        
        self.dataset_loader = DatasetLoader()
    
    def evaluate_spider(
        self,
        split: str = "validation",
        output_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on Spider benchmark.
        
        Args:
            split: Dataset split to evaluate
            output_file: Optional file to save predictions
            
        Returns:
            Dictionary of metrics
        """
        print(f"Evaluating on Spider {split} set...")
        
        # Load Spider dataset
        examples = self.dataset_loader.load_spider(split=split)
        
        # Run inference
        predictions = self._predict_batch(examples)
        
        # Extract references
        references = [ex.sql for ex in examples]
        db_ids = [ex.db_id for ex in examples]
        
        # Compute metrics
        metrics = compute_metrics(
            predictions=predictions,
            references=references,
            db_ids=db_ids if self.sql_executor else None,
            sql_executor=self.sql_executor,
        )
        
        # Save predictions if requested
        if output_file:
            self._save_predictions(examples, predictions, output_file)
        
        return metrics
    
    def evaluate_wikisql(
        self,
        split: str = "validation",
        output_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on WikiSQL benchmark.
        
        Args:
            split: Dataset split
            output_file: Optional file to save predictions
            
        Returns:
            Dictionary of metrics
        """
        print(f"Evaluating on WikiSQL {split} set...")
        
        examples = self.dataset_loader.load_wikisql(split=split)
        predictions = self._predict_batch(examples)
        references = [ex.sql for ex in examples]
        db_ids = [ex.db_id for ex in examples]
        
        metrics = compute_metrics(
            predictions=predictions,
            references=references,
            db_ids=db_ids if self.sql_executor else None,
            sql_executor=self.sql_executor,
        )
        
        if output_file:
            self._save_predictions(examples, predictions, output_file)
        
        return metrics
    
    def evaluate_bird(
        self,
        split: str = "validation",
        output_file: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate on BIRD-SQL benchmark.
        
        Args:
            split: Dataset split
            output_file: Optional file to save predictions
            
        Returns:
            Dictionary of metrics
        """
        print(f"Evaluating on BIRD-SQL {split} set...")
        
        examples = self.dataset_loader.load_bird_sql(split=split)
        
        if not examples:
            print("BIRD-SQL dataset not available")
            return {}
        
        predictions = self._predict_batch(examples)
        references = [ex.sql for ex in examples]
        db_ids = [ex.db_id for ex in examples]
        
        metrics = compute_metrics(
            predictions=predictions,
            references=references,
            db_ids=db_ids if self.sql_executor else None,
            sql_executor=self.sql_executor,
        )
        
        if output_file:
            self._save_predictions(examples, predictions, output_file)
        
        return metrics
    
    def evaluate_all(
        self,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all benchmarks.
        
        Args:
            output_dir: Optional directory to save predictions
            
        Returns:
            Dictionary mapping benchmark names to metrics
        """
        results = {}
        
        output_dir = Path(output_dir) if output_dir else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Spider
        try:
            spider_output = str(output_dir / "spider_predictions.json") if output_dir else None
            results["spider"] = self.evaluate_spider(output_file=spider_output)
        except Exception as e:
            print(f"Spider evaluation failed: {e}")
            results["spider"] = {}
        
        # WikiSQL
        try:
            wikisql_output = str(output_dir / "wikisql_predictions.json") if output_dir else None
            results["wikisql"] = self.evaluate_wikisql(output_file=wikisql_output)
        except Exception as e:
            print(f"WikiSQL evaluation failed: {e}")
            results["wikisql"] = {}
        
        # BIRD-SQL
        try:
            bird_output = str(output_dir / "bird_predictions.json") if output_dir else None
            results["bird"] = self.evaluate_bird(output_file=bird_output)
        except Exception as e:
            print(f"BIRD evaluation failed: {e}")
            results["bird"] = {}
        
        return results
    
    def _predict_batch(
        self,
        examples: List[Text2SQLExample],
        batch_size: int = 8,
    ) -> List[str]:
        """
        Generate predictions for batch of examples.
        
        Args:
            examples: List of examples
            batch_size: Batch size for inference
            
        Returns:
            List of predicted SQL queries
        """
        predictions = []
        
        for i in tqdm(range(0, len(examples), batch_size), desc="Generating predictions"):
            batch_examples = examples[i:i + batch_size]
            
            batch_predictions = self._predict_single_batch(batch_examples)
            predictions.extend(batch_predictions)
        
        return predictions
    
    def _predict_single_batch(
        self,
        examples: List[Text2SQLExample],
    ) -> List[str]:
        """
        Predict for single batch.
        
        Args:
            examples: Batch of examples
            
        Returns:
            List of predictions
        """
        # Prepare inputs
        inputs = []
        for ex in examples:
            encoded = self.tokenizer.encode_example(
                question=ex.question,
                schema=ex.schema,
                sql=None,  # No target for inference
            )
            inputs.append(encoded)
        
        # Batch encode
        batch = self.tokenizer.batch_encode(
            [{"question": ex.question, "schema": ex.schema} for ex in examples],
            max_length=2048,
        )
        
        # Generate predictions
        # This would use the generator - simplified here
        predictions = []
        for ex in examples:
            # Placeholder - actual generation would use beam search
            predictions.append("SELECT * FROM table")  # Dummy prediction
        
        return predictions
    
    def _save_predictions(
        self,
        examples: List[Text2SQLExample],
        predictions: List[str],
        output_file: str,
    ):
        """
        Save predictions to file.
        
        Args:
            examples: Original examples
            predictions: Predicted SQL queries
            output_file: Output file path
        """
        output_data = []
        
        for ex, pred in zip(examples, predictions):
            output_data.append({
                "db_id": ex.db_id,
                "question": ex.question,
                "gold_sql": ex.sql,
                "predicted_sql": pred,
                "source": ex.source,
            })
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Saved predictions to {output_file}")


def run_benchmark(
    model_path: str,
    benchmarks: List[str],
    db_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to run benchmarks.
    
    Args:
        model_path: Path to trained model
        benchmarks: List of benchmarks to run
        db_dir: Directory containing database files
        output_dir: Directory to save results
        
    Returns:
        Dictionary of results
    """
    # Load model
    from ..models.text2sql_model import VantageModel
    from ..data.preprocessing import Tokenizer
    
    print(f"Loading model from {model_path}")
    model = VantageModel.from_pretrained(model_path)
    tokenizer = Tokenizer.from_pretrained(model_path)
    
    # Create SQL executor if db_dir provided
    sql_executor = None
    if db_dir:
        sql_executor = SQLExecutor(db_dir=db_dir, timeout=5.0)
    
    # Create benchmark
    benchmark = Benchmark(
        model=model,
        tokenizer=tokenizer,
        sql_executor=sql_executor,
    )
    
    # Run benchmarks
    results = {}
    
    for benchmark_name in benchmarks:
        if benchmark_name == "spider":
            results["spider"] = benchmark.evaluate_spider(
                output_file=f"{output_dir}/spider_predictions.json" if output_dir else None
            )
        elif benchmark_name == "wikisql":
            results["wikisql"] = benchmark.evaluate_wikisql(
                output_file=f"{output_dir}/wikisql_predictions.json" if output_dir else None
            )
        elif benchmark_name == "bird":
            results["bird"] = benchmark.evaluate_bird(
                output_file=f"{output_dir}/bird_predictions.json" if output_dir else None
            )
    
    # Print results
    print("\n" + "="*50)
    print("BENCHMARK RESULTS")
    print("="*50)
    
    for benchmark_name, metrics in results.items():
        print(f"\n{benchmark_name.upper()}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_dir}/results.json")
    
    return results
