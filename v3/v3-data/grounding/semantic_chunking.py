#!/usr/bin/env python3
"""
Semantic Chunking Module for ACDEK v3 Grounding Engine

This module provides functionality for breaking reference texts into 
meaningful semantic chunks for more precise matching and factuality scoring.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Optional
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
import sys

# Add parent directories to path to enable imports
parent_dir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))

# Try to import embedding functions from appropriate locations
try:
    from storage.vector_store.embed_utils import get_embedding, cosine_similarity, semantic_search
except ImportError:
    try:
        from v3.storage.vector_store.embed_utils import get_embedding, cosine_similarity, semantic_search
    except ImportError:
        logging.warning("Unable to import embedding utilities. Some functionality may be limited.")
        
        # Define fallback functions for testing
        def get_embedding(text: str) -> List[float]:
            """Fallback embedding function."""
            return [0.0] * 384  # Default embedding size
            
        def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
            """Fallback similarity function."""
            return 0.5  # Default similarity
            
        def semantic_search(query: str, candidates: List[str], **kwargs) -> List[Tuple[int, str, float, Any]]:
            """Fallback semantic search function."""
            return [(0, candidates[0] if candidates else "", 0.5, None)]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {str(e)}. Sentence tokenization may be limited.")

class SemanticChunker:
    """
    Handles semantic chunking of text documents for more precise reference matching.
    """
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 500, overlap: int = 50):
        """
        Initialize the semantic chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            overlap: Character overlap between adjacent chunks
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
    
    def chunk_by_sentences(self, text: str, target_sentences: int = 3) -> List[str]:
        """
        Split text into chunks based on sentence boundaries.
        
        Args:
            text: Text to chunk
            target_sentences: Target number of sentences per chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # Tokenize into sentences
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"Error tokenizing sentences: {str(e)}. Falling back to basic splitting.")
            # Simple sentence splitting fallback
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return [text]
            
        # Create chunks with target number of sentences
        chunks = []
        for i in range(0, len(sentences), target_sentences):
            end_idx = min(i + target_sentences, len(sentences))
            chunk = ' '.join(sentences[i:end_idx])
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def chunk_by_semantic_boundaries(self, text: str) -> List[str]:
        """
        Split text into chunks based on semantic boundaries.
        Attempts to detect paragraph breaks and thematic shifts.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
            
        # First split by paragraph boundaries
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Then process each paragraph according to size
        chunks = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            # If paragraph is within size limits, keep it as is
            if len(para) >= self.min_chunk_size and len(para) <= self.max_chunk_size:
                chunks.append(para)
            elif len(para) < self.min_chunk_size:
                # For small paragraphs, combine with previous chunk if possible
                if chunks and len(chunks[-1]) + len(para) <= self.max_chunk_size:
                    chunks[-1] += " " + para
                else:
                    chunks.append(para)
            else:
                # For large paragraphs, split by sentences
                sent_chunks = self.chunk_by_sentences(
                    para, 
                    target_sentences=max(1, int((self.max_chunk_size/2) / 100))
                )
                chunks.extend(sent_chunks)
        
        return chunks
    
    def get_semantic_coherence(self, chunks: List[str]) -> List[float]:
        """
        Calculate the semantic coherence of each chunk.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of coherence scores (higher is more coherent)
        """
        if not chunks:
            return []
            
        # Get embeddings for all chunks
        try:
            embeddings = [get_embedding(chunk) for chunk in chunks]
            
            # Calculate similarity between consecutive chunks
            coherence_scores = []
            for i in range(len(embeddings) - 1):
                similarity = cosine_similarity(embeddings[i], embeddings[i+1])
                coherence_scores.append(similarity)
                
            # Add a default score for the last chunk
            coherence_scores.append(0.7)
            
            return coherence_scores
        except Exception as e:
            logger.warning(f"Error calculating chunk coherence: {str(e)}")
            return [0.7] * len(chunks)  # Default coherence
    
    def merge_low_coherence_chunks(self, chunks: List[str], coherence_scores: List[float], threshold: float = 0.7) -> List[str]:
        """
        Merge chunks with low coherence to improve semantic integrity.
        
        Args:
            chunks: List of text chunks
            coherence_scores: List of coherence scores
            threshold: Coherence threshold for merging
            
        Returns:
            List of merged chunks
        """
        if not chunks or len(chunks) <= 1:
            return chunks
            
        merged_chunks = [chunks[0]]
        
        for i in range(1, len(chunks)):
            # If coherence with previous chunk is high or current chunk is too small, merge
            if coherence_scores[i-1] >= threshold or len(chunks[i]) < self.min_chunk_size:
                if len(merged_chunks[-1]) + len(chunks[i]) <= self.max_chunk_size:
                    merged_chunks[-1] += " " + chunks[i]
                else:
                    merged_chunks.append(chunks[i])
            else:
                merged_chunks.append(chunks[i])
        
        return merged_chunks
    
    def chunk_text(self, text: str, optimize_coherence: bool = True) -> List[Dict[str, Any]]:
        """
        Main method to chunk text with coherence optimization.
        
        Args:
            text: Text to chunk
            optimize_coherence: Whether to optimize for semantic coherence
            
        Returns:
            List of dictionaries with chunk text and metadata
        """
        if not text:
            return []
        
        # Initial chunking by semantic boundaries
        initial_chunks = self.chunk_by_semantic_boundaries(text)
        
        # If requested, optimize coherence
        if optimize_coherence and len(initial_chunks) > 1:
            coherence_scores = self.get_semantic_coherence(initial_chunks)
            final_chunks = self.merge_low_coherence_chunks(initial_chunks, coherence_scores)
        else:
            final_chunks = initial_chunks
        
        # Create result with metadata
        result = []
        for i, chunk in enumerate(final_chunks):
            result.append({
                "id": f"chunk_{i+1}",
                "text": chunk,
                "position": i,
                "char_length": len(chunk),
                "word_count": len(chunk.split())
            })
        
        return result

def chunk_reference_data(reference_data: List[Dict[str, Any]], chunker: Optional[SemanticChunker] = None) -> List[Dict[str, Any]]:
    """
    Process a list of reference data items and chunk their text fields.
    
    Args:
        reference_data: List of reference data dictionaries
        chunker: SemanticChunker instance (created if None)
        
    Returns:
        List of dictionaries with chunked text fields
    """
    if not chunker:
        chunker = SemanticChunker()
    
    chunked_data = []
    
    for ref_item in reference_data:
        ref_id = ref_item.get("id", "unknown")
        ref_text = ref_item.get("text", "")
        
        if not ref_text:
            logger.warning(f"Empty text field in reference data item {ref_id}")
            continue
        
        # Chunk the text
        chunks = chunker.chunk_text(ref_text)
        
        # Create new reference items for each chunk
        for chunk in chunks:
            chunked_item = ref_item.copy()  # Copy all original fields
            chunked_item["id"] = f"{ref_id}_{chunk['id']}"
            chunked_item["text"] = chunk["text"]
            chunked_item["parent_id"] = ref_id
            chunked_item["chunk_metadata"] = {
                "position": chunk["position"],
                "char_length": chunk["char_length"],
                "word_count": chunk["word_count"]
            }
            chunked_data.append(chunked_item)
    
    return chunked_data

def retrieve_reference_chunks(prompt: str, domain: str, all_chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Retrieve most relevant chunks from chunked reference data.
    
    Args:
        prompt: Query prompt
        domain: Domain name
        all_chunks: List of all reference chunks
        top_k: Number of top chunks to return
        
    Returns:
        List of most relevant reference chunks
    """
    if not prompt or not all_chunks:
        return []
    
    # Extract chunk texts for semantic search
    chunk_texts = [chunk.get("text", "") for chunk in all_chunks]
    
    # Perform semantic search
    try:
        search_results = semantic_search(
            query=prompt,
            candidates=chunk_texts,
            threshold=0.5,  # Lower threshold to ensure results
            max_results=top_k
        )
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []
    
    # Extract matched chunks
    matched_chunks = []
    for result in search_results:
        idx, _, score, _ = result
        chunk = all_chunks[idx].copy()
        chunk["similarity"] = score
        matched_chunks.append(chunk)
    
    return matched_chunks

def score_factuality_with_chunks(response: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Score factuality using multiple reference chunks with weighted scoring.
    
    Args:
        response: Generated text to evaluate
        chunks: List of reference chunks with similarity scores
        
    Returns:
        Dictionary with factuality score and metadata
    """
    if not response or not chunks:
        return {
            "score": 0.0,
            "has_contradiction": False,
            "chunk_scores": [],
            "error": "Empty input"
        }
    
    # Calculate individual scores for each chunk
    chunk_scores = []
    total_weight = 0.0
    weighted_sum = 0.0
    
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        chunk_similarity = chunk.get("similarity", 0.0)
        
        # Skip empty chunks
        if not chunk_text:
            continue
        
        # Calculate weight based on similarity to query
        weight = max(0.1, chunk_similarity)  # Ensure minimum weight
        
        # Score against this chunk
        try:
            # Get embeddings
            resp_emb = get_embedding(response)
            chunk_emb = get_embedding(chunk_text)
            
            # Calculate similarity
            similarity = cosine_similarity(resp_emb, chunk_emb)
            
            # Add to weighted sum
            weighted_sum += similarity * weight
            total_weight += weight
            
            # Add to chunk scores
            chunk_scores.append({
                "chunk_id": chunk.get("id", "unknown"),
                "similarity": similarity,
                "weight": weight
            })
        except Exception as e:
            logger.warning(f"Error scoring against chunk {chunk.get('id', 'unknown')}: {str(e)}")
    
    # Calculate final weighted score
    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0
    
    # Check for contradictions (placeholder - to be enhanced)
    has_contradiction = False
    contradiction_confidence = 0.0
    contradiction_reason = ""
    
    return {
        "score": final_score,
        "has_contradiction": has_contradiction,
        "contradiction_confidence": contradiction_confidence,
        "contradiction_reason": contradiction_reason,
        "chunk_scores": chunk_scores,
        "chunks_used": len(chunk_scores),
        "chunk_weights_sum": total_weight
    }

# Test function
def test_chunking(text: str):
    """Test the chunking functionality with provided text."""
    chunker = SemanticChunker()
    chunks = chunker.chunk_text(text)
    
    print(f"Original text length: {len(text)} characters")
    print(f"Generated {len(chunks)} chunks:")
    
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1} ({chunk['char_length']} chars, {chunk['word_count']} words):")
        print(f"{chunk['text'][:100]}...")

if __name__ == "__main__":
    # Simple test when run directly
    test_text = """
    The Federal Reserve primarily controls inflation through monetary policy tools such as setting the federal funds rate, 
    conducting open market operations, and adjusting reserve requirements for banks. By raising interest rates, the Fed can 
    slow economic growth and reduce inflation by making borrowing more expensive, which reduces spending and investment. 
    Conversely, lowering interest rates can stimulate economic activity when inflation is too low.
    
    When inflation rises above the Fed's target (generally 2%), the FOMC typically increases the federal funds rate. 
    This is the interest rate at which depository institutions lend reserve balances to other depository institutions overnight. 
    This increase cascades through the economy, affecting everything from mortgage rates to credit card interest.
    
    Open market operations involve the buying and selling of government securities to control the money supply. 
    When the Fed sells securities, it reduces the money supply and helps combat inflation. When it buys securities, 
    it increases the money supply, which can combat deflation.
    """
    
    test_chunking(test_text)