"""
RAGAS Evaluation - Faithfulness, Answer Relevancy, and Context Precision.
RAGAS (RAG Assessment) is the standard evaluation framework for RAG systems.
"""

import logging
import importlib.util
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """Results from RAGAS evaluation."""

    faithfulness: float
    answer_relevancy: float
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    num_samples: int = 0

    @property
    def overall_score(self) -> float:
        """Harmonic mean of available scores."""
        scores = [self.faithfulness, self.answer_relevancy]
        if self.context_precision:
            scores.append(self.context_precision)
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    def to_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
            "overall_score": self.overall_score,
            "num_samples": self.num_samples,
        }


class RAGASEvaluator:
    """
    RAGAS-based RAG evaluation.
    """

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._ragas_available = self._check_ragas()

    def evaluate(
        self,
        questions: list[str],
        answers: list[str],
        contexts: list[list[str]],
        ground_truths: Optional[list[str]] = None,
    ) -> RAGASResult:
        """Run RAGAS evaluation on a set of QA pairs."""
        if self._ragas_available:
            return self._run_ragas(questions, answers, contexts, ground_truths)
        else:
            logger.warning(
                "RAGAS not available. Using lightweight approximation metrics."
            )
            return self._lightweight_eval(questions, answers, contexts)

    def _run_ragas(self, questions, answers, contexts, ground_truths) -> RAGASResult:
        """Run official RAGAS evaluation."""
        try:
            # Local imports here ensure we only depend on RAGAS when necessary
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import faithfulness, answer_relevancy, context_precision
            from datasets import Dataset

            data = {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
            }
            if ground_truths:
                data["ground_truth"] = ground_truths

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy]
            if ground_truths:
                metrics.append(context_precision)

            result = ragas_evaluate(dataset=dataset, metrics=metrics)
            scores = result.to_pandas()

            return RAGASResult(
                faithfulness=round(float(scores["faithfulness"].mean()), 4),
                answer_relevancy=round(float(scores["answer_relevancy"].mean()), 4),
                context_precision=(
                    round(float(scores["context_precision"].mean()), 4)
                    if ground_truths
                    else None
                ),
                num_samples=len(questions),
            )
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            return self._lightweight_eval(questions, answers, contexts)

    def _lightweight_eval(self, questions, answers, contexts) -> RAGASResult:
        """
        Lightweight approximation without RAGAS dependency.
        """
        import numpy as np

        faithfulness_scores = []
        relevancy_scores = []

        for q, a, ctx_list in zip(questions, answers, contexts):
            # Faithfulness: how much of answer appears in context
            ctx_text = " ".join(ctx_list).lower()
            answer_words = set(a.lower().split())
            ctx_words = set(ctx_text.split())
            overlap = len(answer_words & ctx_words) / max(len(answer_words), 1)
            faithfulness_scores.append(min(overlap * 2, 1.0))

            # Answer relevancy: word overlap between question and answer
            q_words = set(q.lower().split())
            a_words = set(a.lower().split())
            relevancy = len(q_words & a_words) / max(len(q_words), 1)
            relevancy_scores.append(relevancy)

        return RAGASResult(
            faithfulness=round(float(np.mean(faithfulness_scores)), 4),
            answer_relevancy=round(float(np.mean(relevancy_scores)), 4),
            num_samples=len(questions),
        )

    def _check_ragas(self) -> bool:
        """Check if RAGAS is available without importing it at the module level."""
        return importlib.util.find_spec("ragas") is not None
