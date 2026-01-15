# semantic_evaluator.py
import sys
sys.path.insert(0, "gepa/src")

from sentence_transformers import SentenceTransformer
from gepa.adapters.default_adapter.default_adapter import EvaluationResult
import numpy as np


class SemanticSimilarityEvaluator:
    """Evaluates generated descriptions using embedding cosine similarity."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")
        self.embedder = SentenceTransformer(model_name)
        print("Embedding model loaded.")

    def __call__(self, data, response):
        """
        Args:
            data: {"input": SMILES, "answer": ground_truth_description, "additional_context": {}}
            response: Gemini's generated description
        Returns:
            EvaluationResult(score, feedback, objective_scores)
        """
        expected = data["answer"]

        # Handle empty responses
        if not response or not response.strip():
            return EvaluationResult(
                score=0.0,
                feedback="Empty response generated. The model should produce a description.",
                objective_scores=None
            )

        # Compute embeddings
        emb_gen = self.embedder.encode(response, convert_to_numpy=True)
        emb_exp = self.embedder.encode(expected, convert_to_numpy=True)

        # Cosine similarity
        norm_gen = np.linalg.norm(emb_gen)
        norm_exp = np.linalg.norm(emb_exp)

        if norm_gen == 0 or norm_exp == 0:
            score = 0.0
        else:
            score = float(np.dot(emb_gen, emb_exp) / (norm_gen * norm_exp))
            score = max(0.0, score)  # Clamp to [0, 1]

        # Generate feedback for reflection
        if score > 0.8:
            feedback = f"Good description (similarity={score:.2f}). Generated captures key concepts."
        elif score > 0.5:
            feedback = (
                f"Partial match (similarity={score:.2f}). "
                f"Expected description contains: '{expected}...'. "
                "Focus on molecule class, functional groups, and biological roles."
            )
        else:
            feedback = (
                f"Poor match (similarity={score:.2f}). "
                f"Expected: '{expected}'. "
                "The description should identify the molecule type, structural features, and any biological roles."
            )

        return EvaluationResult(score=score, feedback=feedback, objective_scores=None)
