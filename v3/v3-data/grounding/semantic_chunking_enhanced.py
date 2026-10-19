#!/usr/bin/env python3
"""
Enhanced Semantic Chunking Module for ACDEK v3 Grounding Engine (Phase 3)

This module provides advanced functionality for breaking reference texts into 
meaningful semantic chunks for more precise matching and factuality scoring.
Improvements include:
- Hierarchical chunking
- Optimized chunking parameters
- Enhanced semantic coherence
- Improved chunk caching
- Better fallback mechanisms
"""

import re
import logging
import json
import os
import hashlib
from typing import List, Dict, Any, Tuple, Optional, Union, Set
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from pathlib import Path
import sys
import time

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

# Download NLTK resources if needed
def ensure_nltk_resources():
    """Ensure required NLTK resources are available."""
    resources = [
        ('punkt', 'tokenizers/punkt'),
        ('stopwords', 'corpora/stopwords'),
        ('averaged_perceptron_tagger', 'taggers/averaged_perceptron_tagger')
    ]
    
    for resource, path in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
                logger.info(f"Downloaded NLTK resource: {resource}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {resource}: {str(e)}")

# Initialize NLTK resources
ensure_nltk_resources()

class ChunkCache:
    """
    Cache for chunked texts to improve performance.
    Implements a simple LRU (Least Recently Used) strategy.
    """
    
    def __init__(self, max_size: int = 100, cache_dir: Optional[str] = None):
        """
        Initialize chunk cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            cache_dir: Directory to persist cache (if None, in-memory only)
        """
        self.max_size = max_size
        self.cache = {}  # text_hash -> {'chunks': chunks, 'timestamp': access_time}
        self.cache_dir = cache_dir
        
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()
            
    def _compute_hash(self, text: str) -> str:
        """Compute hash of text for cache key."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _load_cache(self):
        """Load cache from disk if available."""
        cache_file = os.path.join(self.cache_dir, 'chunk_cache.json')
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} chunk entries from cache")
            except Exception as e:
                logger.error(f"Error loading chunk cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk if directory is configured."""
        if not self.cache_dir:
            return
            
        cache_file = os.path.join(self.cache_dir, 'chunk_cache.json')
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.cache, f)
        except Exception as e:
            logger.error(f"Error saving chunk cache: {e}")
    
    def get(self, text: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get chunks from cache if available.
        
        Args:
            text: Original text to lookup
            
        Returns:
            Cached chunks if available, None otherwise
        """
        text_hash = self._compute_hash(text)
        
        if text_hash in self.cache:
            # Update timestamp
            self.cache[text_hash]['timestamp'] = time.time()
            return self.cache[text_hash]['chunks']
            
        return None
    
    def put(self, text: str, chunks: List[Dict[str, Any]]):
        """
        Add chunks to cache.
        
        Args:
            text: Original text
            chunks: Processed chunks
        """
        text_hash = self._compute_hash(text)
        
        # If cache is full, remove least recently used entry
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['timestamp'])
            del self.cache[oldest_key]
        
        # Add to cache
        self.cache[text_hash] = {
            'chunks': chunks,
            'timestamp': time.time()
        }
        
        # Save to disk if configured
        self._save_cache()

class SemanticChunkerEnhanced:
    """
    Enhanced module for semantic chunking of text documents with improved
    precision and optimization.
    """
    
    def __init__(self, 
                 min_chunk_size: int = 100, 
                 max_chunk_size: int = 500, 
                 overlap: int = 50,
                 cache_dir: Optional[str] = None):
        """
        Initialize the enhanced semantic chunker.
        
        Args:
            min_chunk_size: Minimum chunk size in characters
            max_chunk_size: Maximum chunk size in characters
            overlap: Character overlap between adjacent chunks
            cache_dir: Directory for chunk cache (None for no caching)
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        
        # Domain-specific optimal parameters
        self.domain_params = {
            'medical': {'min_size': 150, 'max_size': 600, 'overlap': 75},
            'legal': {'min_size': 200, 'max_size': 700, 'overlap': 100},
            'financial': {'min_size': 120, 'max_size': 550, 'overlap': 60}
        }
        
        # Initialize cache
        self.cache = ChunkCache(max_size=200, cache_dir=cache_dir)
        
        # Try to load stopwords for enhanced coherence
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set(['the', 'and', 'is', 'in', 'it', 'to', 'of', 'for', 'a', 'with'])
    
    def set_domain_parameters(self, domain: str):
        """
        Set chunking parameters optimized for a specific domain.
        
        Args:
            domain: Domain name (medical, legal, financial)
        """
        domain = domain.lower()
        if domain in self.domain_params:
            params = self.domain_params[domain]
            self.min_chunk_size = params['min_size']
            self.max_chunk_size = params['max_size']
            self.overlap = params['overlap']
            logger.info(f"Set chunking parameters for domain: {domain}")
        else:
            logger.warning(f"No optimized parameters for domain: {domain}. Using defaults.")
    
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
    
    def chunk_hierarchically(self, text: str) -> List[Dict[str, Any]]:
        """
        New method that creates a hierarchical chunking structure.
        Creates three levels: document, section, and chunk.
        
        Args:
            text: Text to chunk
            
        Returns:
            List of chunk dictionaries with hierarchical metadata
        """
        if not text:
            return []
            
        # Level 1: Document - the entire text
        document_emb = get_embedding(text)
        
        # Level 2: Section - split by major semantic breaks
        # Look for section patterns like headers, numbered sections, etc.
        section_patterns = [
            r'\n\s*#+\s+[A-Z]',  # Markdown headers
            r'\n\s*[A-Z][A-Z\s]+\n',  # ALL CAPS headers
            r'\n\s*\d+\.\s+[A-Z]',  # Numbered sections
            r'\n\s*[A-Z][a-z]+\s+\d+',  # Section/Article headings
        ]
        
        # Apply section splitting patterns
        current_text = text
        for pattern in section_patterns:
            if re.search(pattern, current_text):
                current_text = re.split(pattern, current_text)
                current_text = [text if i == 0 else pattern[4:] + text for i, text in enumerate(current_text)]
                break
        
        # If no sections found, treat whole text as one section
        if isinstance(current_text, str):
            sections = [current_text]
        else:
            sections = current_text
        
        # Level 3: Chunk - for each section, create smaller chunks
        all_chunks = []
        section_id = 0
        
        for section in sections:
            section_id += 1
            section_text = section.strip()
            if not section_text:
                continue
                
            # Get section embedding
            section_emb = get_embedding(section_text)
            
            # Create chunks for this section
            section_chunks = self.chunk_by_semantic_boundaries(section_text)
            
            chunk_id = 0
            for chunk_text in section_chunks:
                chunk_id += 1
                
                # Create chunk with hierarchical metadata
                chunk = {
                    "id": f"chunk_{section_id}_{chunk_id}",
                    "text": chunk_text,
                    "position": chunk_id,
                    "section_id": section_id,
                    "char_length": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                    "similarity_to_section": cosine_similarity(get_embedding(chunk_text), section_emb),
                    "similarity_to_document": cosine_similarity(get_embedding(chunk_text), document_emb)
                }
                
                all_chunks.append(chunk)
        
        return all_chunks
    
    def get_semantic_coherence(self, chunks: List[str]) -> List[float]:
        """
        Enhanced method to calculate the semantic coherence of each chunk.
        
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
            
            # Enhanced method: calculate similarity of each chunk to both
            # its preceding and following chunks
            for i in range(len(embeddings)):
                # Similarity to previous chunk
                prev_sim = 0.0
                if i > 0:
                    prev_sim = cosine_similarity(embeddings[i], embeddings[i-1])
                
                # Similarity to next chunk
                next_sim = 0.0
                if i < len(embeddings) - 1:
                    next_sim = cosine_similarity(embeddings[i], embeddings[i+1])
                
                # Weighted coherence score
                if i == 0:  # First chunk
                    coherence = next_sim
                elif i == len(embeddings) - 1:  # Last chunk
                    coherence = prev_sim
                else:  # Middle chunks
                    coherence = (prev_sim + next_sim) / 2.0
                
                coherence_scores.append(coherence)
            
            return coherence_scores
            
        except Exception as e:
            logger.warning(f"Error calculating enhanced chunk coherence: {str(e)}")
            return [0.7] * len(chunks)  # Default coherence
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        Extract the most important keywords from text.
        
        Args:
            text: Input text
            top_n: Number of keywords to extract
            
        Returns:
            List of keywords
        """
        try:
            # Tokenize and convert to lowercase
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and non-alphabetic tokens
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in self.stopwords and len(word) > 3]
            
            # Count frequency
            word_freq = {}
            for word in filtered_tokens:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Sort by frequency
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            
            # Return top N keywords
            keywords = [word for word, freq in sorted_words[:top_n]]
            return keywords
            
        except Exception as e:
            logger.warning(f"Error extracting keywords: {str(e)}")
            return []
    
    def merge_low_coherence_chunks(self, chunks: List[str], coherence_scores: List[float], threshold: float = 0.7) -> List[str]:
        """
        Enhanced method to merge chunks with low coherence to improve semantic integrity.
        
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
    
    def chunk_text(self, text: str, optimize_coherence: bool = True, hierarchical: bool = True,
                  domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Enhanced main method to chunk text with multiple improvements.
        
        Args:
            text: Text to chunk
            optimize_coherence: Whether to optimize for semantic coherence
            hierarchical: Whether to use hierarchical chunking
            domain: Domain for optimized parameters
            
        Returns:
            List of dictionaries with chunk text and metadata
        """
        if not text:
            return []
        
        # Check cache first
        cached_chunks = self.cache.get(text)
        if cached_chunks:
            logger.debug("Using cached chunks")
            return cached_chunks
        
        # Set domain-specific parameters if provided
        if domain:
            self.set_domain_parameters(domain)
        
        # Use hierarchical chunking if requested
        if hierarchical:
            chunks = self.chunk_hierarchically(text)
            self.cache.put(text, chunks)
            return chunks
        
        # Otherwise use standard chunking with coherence optimization
        initial_chunks = self.chunk_by_semantic_boundaries(text)
        
        # If requested, optimize coherence
        if optimize_coherence and len(initial_chunks) > 1:
            coherence_scores = self.get_semantic_coherence(initial_chunks)
            final_chunks = self.merge_low_coherence_chunks(initial_chunks, coherence_scores)
        else:
            final_chunks = initial_chunks
        
        # Create result with enhanced metadata
        result = []
        for i, chunk in enumerate(final_chunks):
            # Extract keywords for each chunk
            keywords = self.extract_keywords(chunk)
            
            result.append({
                "id": f"chunk_{i+1}",
                "text": chunk,
                "position": i,
                "char_length": len(chunk),
                "word_count": len(chunk.split()),
                "keywords": keywords
            })
        
        # Cache result
        self.cache.put(text, result)
        
        return result

def chunk_reference_data_enhanced(reference_data: List[Dict[str, Any]], 
                                 chunker: Optional[SemanticChunkerEnhanced] = None,
                                 domain: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Enhanced process for chunking a list of reference data items.
    
    Args:
        reference_data: List of reference data dictionaries
        chunker: SemanticChunkerEnhanced instance (created if None)
        domain: Domain for optimized chunking
        
    Returns:
        List of dictionaries with chunked text fields
    """
    if not chunker:
        chunker = SemanticChunkerEnhanced()
    
    chunked_data = []
    
    for ref_item in reference_data:
        ref_id = ref_item.get("id", "unknown")
        ref_text = ref_item.get("text", "")
        
        if not ref_text:
            logger.warning(f"Empty text field in reference data item {ref_id}")
            continue
        
        # Chunk the text
        chunks = chunker.chunk_text(ref_text, domain=domain)
        
        # Create new reference items for each chunk
        for chunk in chunks:
            chunked_item = ref_item.copy()  # Copy all original fields
            chunked_item["id"] = f"{ref_id}_{chunk['id']}"
            chunked_item["text"] = chunk["text"]
            chunked_item["parent_id"] = ref_id
            
            # Add extended metadata
            metadata = {k: v for k, v in chunk.items() if k != 'text'}
            
            # If reference data has source or definition, add keywords from those too
            if 'source' in ref_item:
                metadata['source'] = ref_item['source']
            
            if 'definition' in ref_item:
                # For enhanced reference format
                metadata['keywords'] = chunker.extract_keywords(ref_item['definition'])
                
            chunked_item["chunk_metadata"] = metadata
            chunked_data.append(chunked_item)
    
    return chunked_data

def retrieve_reference_chunks_enhanced(prompt: str, 
                                      domain: str, 
                                      all_chunks: List[Dict[str, Any]], 
                                      top_k: int = 5,
                                      min_similarity: float = 0.25) -> List[Dict[str, Any]]:
    """
    Enhanced retrieval of most relevant chunks from chunked reference data.
    
    Args:
        prompt: Query prompt
        domain: Domain name
        all_chunks: List of all reference chunks
        top_k: Number of top chunks to return
        min_similarity: Minimum similarity threshold for inclusion
        
    Returns:
        List of most relevant reference chunks
    """
    if not prompt or not all_chunks:
        return []
    
    # Extract prompt keywords for better matching
    chunker = SemanticChunkerEnhanced()
    prompt_keywords = chunker.extract_keywords(prompt, top_n=8)
    
    # Extract chunk texts for semantic search
    chunk_texts = [chunk.get("text", "") for chunk in all_chunks]
    
    # Perform semantic search
    try:
        search_results = semantic_search(
            query=prompt,
            candidates=chunk_texts,
            threshold=min_similarity,  # Lower threshold than before
            max_results=top_k * 2  # Get more candidates for filtering
        )
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []
    
    # Extract matched chunks
    matched_chunks = []
    
    # Enhanced ranking: consider keyword matches along with overall similarity
    for result in search_results:
        idx, _, score, _ = result
        chunk = all_chunks[idx].copy()
        
        # Basic similarity score from vector similarity
        chunk["similarity"] = score
        
        # Calculate keyword overlap
        chunk_keywords = []
        if "keywords" in chunk:
            chunk_keywords = chunk["keywords"]
        elif "chunk_metadata" in chunk and "keywords" in chunk["chunk_metadata"]:
            chunk_keywords = chunk["chunk_metadata"]["keywords"]
        
        # Count matching keywords between prompt and chunk
        keyword_matches = sum(1 for kw in prompt_keywords if kw in chunk_keywords)
        keyword_overlap = keyword_matches / max(1, min(len(prompt_keywords), len(chunk_keywords)))
        
        # Adjust score based on keyword overlap
        adjusted_score = score * 0.7 + keyword_overlap * 0.3
        chunk["adjusted_similarity"] = adjusted_score
        
        matched_chunks.append(chunk)
    
    # Sort by adjusted similarity
    matched_chunks.sort(key=lambda x: x.get("adjusted_similarity", 0.0), reverse=True)
    
    # Return top-k after enhanced ranking
    return matched_chunks[:top_k]

def score_factuality_with_chunks_enhanced(response: str, chunks: List[Dict[str, Any]], 
                                         domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Enhanced scoring of factuality using multiple reference chunks with improved weighting.
    
    Args:
        response: Generated text to evaluate
        chunks: List of reference chunks with similarity scores
        domain: Domain for domain-specific scoring adjustments
        
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
    
    # Calculate individual scores for each chunk with improved weighting
    chunk_scores = []
    total_weight = 0.0
    weighted_sum = 0.0
    
    # Domain-specific similarity thresholds
    domain_similarity_thresholds = {
        "medical": 0.6,
        "legal": 0.65,
        "financial": 0.55
    }
    
    # Default threshold
    similarity_threshold = 0.5
    if domain and domain in domain_similarity_thresholds:
        similarity_threshold = domain_similarity_thresholds[domain]
    
    for chunk in chunks:
        chunk_text = chunk.get("text", "")
        
        # Get adjusted similarity if available, otherwise use regular similarity
        chunk_similarity = chunk.get("adjusted_similarity", chunk.get("similarity", 0.0))
        
        # Skip chunks with very low similarity
        if chunk_similarity < similarity_threshold / 2:
            continue
        
        # Skip empty chunks
        if not chunk_text:
            continue
        
        # Calculate weight based on adjusted similarity to query
        base_weight = max(0.1, chunk_similarity)
        
        # Adjust weight based on chunk metadata if available
        metadata = chunk.get("chunk_metadata", {})
        section_sim = metadata.get("similarity_to_section", 1.0)
        doc_sim = metadata.get("similarity_to_document", 1.0)
        
        # Enhanced weighting formula
        weight = base_weight * (0.7 + 0.15 * section_sim + 0.15 * doc_sim)
        
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
                "query_similarity": chunk_similarity,
                "weight": weight
            })
        except Exception as e:
            logger.warning(f"Error scoring against chunk {chunk.get('id', 'unknown')}: {str(e)}")
    
    # Calculate final weighted score
    if total_weight > 0:
        final_score = weighted_sum / total_weight
    else:
        final_score = 0.0
    
    # Enhanced output with more metadata
    result = {
        "score": final_score,
        "chunk_scores": chunk_scores,
        "chunks_used": len(chunk_scores),
        "chunk_weights_sum": total_weight,
        "average_chunk_similarity": sum(c["similarity"] for c in chunk_scores) / max(1, len(chunk_scores)),
        "domain": domain
    }
    
    return result

# Test function
def test_chunking_enhanced(text: str, domain: Optional[str] = None):
    """Test the enhanced chunking functionality with provided text."""
    chunker = SemanticChunkerEnhanced()
    
    if domain:
        chunker.set_domain_parameters(domain)
        
    standard_chunks = chunker.chunk_by_semantic_boundaries(text)
    hierarchical_chunks = chunker.chunk_text(text, hierarchical=True, domain=domain)
    
    print(f"Original text length: {len(text)} characters")
    print(f"\nStandard chunking generated {len(standard_chunks)} chunks")
    print(f"Hierarchical chunking generated {len(hierarchical_chunks)} chunks")
    
    print("\n--- Standard Chunking Results ---")
    for i, chunk in enumerate(standard_chunks):
        print(f"\nChunk {i+1} ({len(chunk)} chars):")
        print(f"{chunk[:100]}...")
    
    print("\n--- Hierarchical Chunking Results ---")
    for i, chunk in enumerate(hierarchical_chunks):
        section_id = chunk.get("section_id", "?")
        print(f"\nChunk {i+1} (Section {section_id}, {chunk['char_length']} chars, {chunk['word_count']} words):")
        print(f"{chunk['text'][:100]}...")
        print(f"Keywords: {', '.join(chunk.get('keywords', []))}")
        print(f"Similarity to section: {chunk.get('similarity_to_section', 'N/A')}")
        print(f"Similarity to document: {chunk.get('similarity_to_document', 'N/A')}")

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
    
    test_chunking_enhanced(test_text, domain="financial")