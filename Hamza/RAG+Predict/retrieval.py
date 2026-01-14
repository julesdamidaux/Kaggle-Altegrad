"""
Retrieval system for finding similar graphs using embeddings.
Implements k-nearest neighbor search with various distance metrics.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import List, Tuple
import config


class GraphRetriever:
    """Retrieval system for finding similar molecular graphs."""
    
    def __init__(self, embeddings: np.ndarray, descriptions: List[str], 
                 graph_ids: List[int], metric: str = "cosine"):
        """
        Initialize retriever with precomputed embeddings.
        
        Args:
            embeddings: Numpy array of shape [num_graphs, embedding_dim]
            descriptions: List of descriptions corresponding to each graph
            graph_ids: List of graph IDs
            metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.embeddings = embeddings
        self.descriptions = descriptions
        self.graph_ids = graph_ids
        self.metric = metric
        
        # Normalize embeddings for cosine similarity
        if metric == "cosine":
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.normalized_embeddings = embeddings / (norms + 1e-8)
        
        print(f"Retriever initialized with {len(embeddings)} graphs using {metric} metric")
    
    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[str], List[float], List[int]]:
        """
        Retrieve k most similar graphs to the query.
        
        Args:
            query_embedding: Query graph embedding [embedding_dim]
            k: Number of neighbors to retrieve
        
        Returns:
            descriptions: List of k descriptions
            distances: List of k similarity scores/distances
            graph_ids: List of k graph IDs
        """
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Compute distances based on metric
        if self.metric == "cosine":
            # Normalize query
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            # Compute cosine similarity (higher is better)
            similarities = cosine_similarity(query_norm, self.normalized_embeddings)[0]
            # Get top-k (highest similarity)
            top_k_indices = np.argsort(similarities)[::-1][:k]
            scores = similarities[top_k_indices]
            
        elif self.metric == "euclidean":
            # Compute Euclidean distances (lower is better)
            distances = euclidean_distances(query_embedding, self.embeddings)[0]
            # Get top-k (lowest distance)
            top_k_indices = np.argsort(distances)[:k]
            scores = distances[top_k_indices]
            
        elif self.metric == "dot":
            # Compute dot product (higher is better)
            dot_products = np.dot(self.embeddings, query_embedding.T).squeeze()
            # Get top-k (highest dot product)
            top_k_indices = np.argsort(dot_products)[::-1][:k]
            scores = dot_products[top_k_indices]
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        # Retrieve corresponding descriptions and IDs
        retrieved_descriptions = [self.descriptions[idx] for idx in top_k_indices]
        retrieved_ids = [self.graph_ids[idx] for idx in top_k_indices]
        
        return retrieved_descriptions, scores.tolist(), retrieved_ids
    
    def batch_retrieve(self, query_embeddings: np.ndarray, k: int = 5) -> List[Tuple[List[str], List[float], List[int]]]:
        """
        Retrieve k most similar graphs for a batch of queries.
        
        Args:
            query_embeddings: Batch of query embeddings [batch_size, embedding_dim]
            k: Number of neighbors to retrieve per query
        
        Returns:
            List of (descriptions, scores, graph_ids) tuples for each query
        """
        results = []
        for query in query_embeddings:
            results.append(self.retrieve(query, k=k))
        return results


def format_retrieved_context(descriptions: List[str], scores: List[float]) -> str:
    """
    Format retrieved descriptions into a context string for generation.
    
    Args:
        descriptions: List of retrieved descriptions
        scores: List of similarity scores
    
    Returns:
        Formatted context string
    """
    context = "Similar molecules and their descriptions:\n\n"
    
    for i, (desc, score) in enumerate(zip(descriptions, scores), 1):
        context += f"{i}. (Similarity: {score:.3f})\n{desc}\n\n"
    
    return context


def create_rag_prompt(query_description: str, retrieved_descriptions: List[str], 
                     scores: List[float]) -> str:
    """
    Create a prompt that includes retrieved context for RAG-based generation.
    
    Args:
        query_description: Instruction/query for the current molecule
        retrieved_descriptions: List of retrieved similar molecule descriptions
        scores: Similarity scores
    
    Returns:
        Complete prompt with context
    """
    context = format_retrieved_context(retrieved_descriptions, scores)
    
    prompt = f"""Given the following similar molecules and their descriptions:

{context}

Now, generate a detailed chemical description for the target molecule following a similar style and level of detail.

Target molecule description:"""
    
    return prompt
