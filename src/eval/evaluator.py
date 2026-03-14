from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from loguru import logger

class EvalMetric(ABC):
    """Base class for evaluation metrics."""
    @abstractmethod
    def score(self, query: str, answer: str, context: List[str]) -> float:
        pass

class FaithfulnessMetric(EvalMetric):
    """Measures how faithful the answer is to the retrieved context."""
    def score(self, query: str, answer: str, context: List[str]) -> float:
        logger.debug(f"Computing faithfulness score for answer: {answer[:50]}...")
        # Placeholder for LLM-based faithfulness check (using Ragas logic)
        return 0.9  # Mock score

class AnswerRelevancyMetric(EvalMetric):
    """Measures how relevant the answer is to the query."""
    def score(self, query: str, answer: str, context: List[str]) -> float:
        logger.debug(f"Computing answer relevancy for query: {query}")
        # Placeholder for LLM-based relevancy check
        return 0.85  # Mock score

class EvaluationSuite:
    """Orchestrates multiple evaluation metrics for a RAG system."""
    
    def __init__(self, metrics: List[EvalMetric] = None):
        self.metrics = metrics or [FaithfulnessMetric(), AnswerRelevancyMetric()]
        
    def evaluate(self, query: str, answer: str, context: List[str]) -> Dict[str, float]:
        """Runs all metrics and returns a score report."""
        logger.info(f"Running evaluation suite for query: {query}")
        results = {}
        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            results[metric_name] = metric.score(query, answer, context)
            
        logger.info(f"Evaluation results: {results}")
        return results

class BatchEvaluator:
    """Handles batch evaluation of multiple query-answer-context triplets."""
    def __init__(self, suite: EvaluationSuite = None):
        self.suite = suite or EvaluationSuite()

    def run_batch(self, dataset: List[Dict[str, Union[str, List[str]]]]) -> List[Dict[str, float]]:
        """
        Runs evaluation on a list of samples.
        Dataset format: [{'query': '...', 'answer': '...', 'context': [...]}]
        """
        batch_results = []
        for sample in dataset:
            res = self.suite.evaluate(sample['query'], sample['answer'], sample['context'])
            batch_results.append(res)
        return batch_results
