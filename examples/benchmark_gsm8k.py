"""
GSM8K Benchmark Evaluation Script for HyperReason Engine
Executes automated test-time compute scaling benchmarks across GSM8K word problems.
"""

from hyper_reason import ReasonEngine, SearchConfig
from hyper_reason.datasets import GSM8KDataset, BenchmarkEvaluator

def run_benchmark():
    print("=" * 70)
    print("      HYPERREASON ENGINE: RUNNING GSM8K BENCHMARK EVALUATION          ")
    print("=" * 70)

    dataset = GSM8KDataset().get_problems()
    print(f"Loaded {len(dataset)} GSM8K evaluation problems.\n")

    config = SearchConfig(num_simulations=24, max_depth=5, prune_kv_cache=True)
    engine = ReasonEngine(config=config)
    evaluator = BenchmarkEvaluator(engine=engine)

    results = evaluator.evaluate_dataset(dataset)

    print("📊 EVALUATION RESULTS SUMMARY:")
    print("-" * 50)
    print(f"Total Evaluated: {results['total_evaluated']}")
    print(f"Correct Answers: {results['correct_answers']}")
    print(f"Pass@1 Accuracy: {results['accuracy_pct']}%")
    print("-" * 50)

    print("\nDetailed Problem Results:")
    for detail in results["details"]:
        status = "✅ PASS" if detail["is_correct"] else "❌ FAIL"
        print(f"  [{status}] ID: {detail['id']} | Target: {detail['target']} | Predicted: {detail['predicted']} (Conf: {detail['confidence']})")

if __name__ == "__main__":
    run_benchmark()
