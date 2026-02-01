"""
Evaluation script for Vantage models
"""

import argparse
from pathlib import Path
import sys
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.benchmark import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Evaluate Vantage text-to-SQL model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=["spider"],
        choices=["spider", "bird", "wikisql", "all"],
        help="Benchmarks to run"
    )
    parser.add_argument(
        "--db_dir",
        type=str,
        default=None,
        help="Directory containing database files (for execution accuracy)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Expand "all" to all benchmarks
    benchmarks = args.benchmarks
    if "all" in benchmarks:
        benchmarks = ["spider", "bird", "wikisql"]
    
    print("="*50)
    print("VANTAGE MODEL EVALUATION")
    print("="*50)
    print(f"\nModel: {args.model_path}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"Output: {args.output_dir}")
    
    if args.db_dir:
        print(f"Databases: {args.db_dir}")
        print("Execution accuracy will be computed")
    else:
        print("No database directory - only exact match will be computed")
    
    print("\n" + "="*50 + "\n")
    
    # Run benchmarks
    results = run_benchmark(
        model_path=args.model_path,
        benchmarks=benchmarks,
        db_dir=args.db_dir,
        output_dir=args.output_dir,
    )
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    
    # Summary
    print("\nSummary:")
    for benchmark, metrics in results.items():
        print(f"\n{benchmark.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.2%}")
            else:
                print(f"  {metric}: {value}")
    
    # Save summary
    summary_path = Path(args.output_dir) / "summary.txt"
    with open(summary_path, 'w') as f:
        f.write("VANTAGE MODEL EVALUATION SUMMARY\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {args.model_path}\n\n")
        
        for benchmark, metrics in results.items():
            f.write(f"{benchmark.upper()}:\n")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"  {metric}: {value:.2%}\n")
                else:
                    f.write(f"  {metric}: {value}\n")
            f.write("\n")
    
    print(f"\nDetailed results saved to: {args.output_dir}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
