import logging
import numpy as np
import os
import pickle
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Union, Callable, Set, Type
from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from registry.registry_types import PluginType
from registry.plugin_registry import register_plugin
from interfaces.vector_alignment_interface import VectorAlignmentInterface
from storage.vector_store.vector_store_interface import VectorStoreInterface
from storage.vector_store.embed_utils import get_embedding, batch_embed, cosine_similarity, DEFAULT_MODEL

logger = logging.getLogger(__name__)


class ModalityType(str, Enum):
    """Enumeration of supported modality types."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    STRUCTURED = "structured"
    MIXED = "mixed"
    CUSTOM = "custom"


@dataclass
class ModalityConfig:
    """Configuration for a specific modality."""
    modality_type: ModalityType
    embedding_model: str  # Model name or identifier
    dimension: int  # Vector dimension
    enabled: bool = True
    similarity_threshold: float = 0.6
    metadata_fields: List[str] = field(default_factory=list)
    custom_handler: Optional[str] = None  # Module.Class path for custom handlers
    config: Dict[str, Any] = field(default_factory=dict)  # Additional config for custom handlers

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "modality_type": self.modality_type.value,
            "embedding_model": self.embedding_model,
            "dimension": self.dimension,
            "enabled": self.enabled,
            "similarity_threshold": self.similarity_threshold,
        }
        if self.metadata_fields:
            result["metadata_fields"] = self.metadata_fields
        if self.custom_handler:
            result["custom_handler"] = self.custom_handler
        if self.config:
            result["config"] = self.config
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModalityConfig':
        """Create from dictionary representation."""
        modality_type = ModalityType(data["modality_type"])
        embedding_model = data["embedding_model"]
        dimension = data["dimension"]
        enabled = data.get("enabled", True)
        similarity_threshold = data.get("similarity_threshold", 0.6)
        metadata_fields = data.get("metadata_fields", [])
        custom_handler = data.get("custom_handler")
        config = data.get("config", {})
        
        return cls(
            modality_type=modality_type,
            embedding_model=embedding_model,
            dimension=dimension,
            enabled=enabled,
            similarity_threshold=similarity_threshold,
            metadata_fields=metadata_fields,
            custom_handler=custom_handler,
            config=config
        )


class ModalityHandler(ABC):
    """
    Base class for modality-specific handlers.
    
    Handlers are responsible for processing different types of content
    and generating appropriate embeddings for them.
    """
    
    def __init__(self, config: ModalityConfig):
        """
        Initialize the modality handler.
        
        Args:
            config: Configuration for this modality
        """
        self.config = config
    
    @abstractmethod
    def create_embedding(self, content: Any) -> Optional[np.ndarray]:
        """
        Create an embedding for the given content.
        
        Args:
            content: Content to embed (type depends on modality)
            
        Returns:
            Embedding vector or None if not applicable
        """
        pass
    
    @abstractmethod
    def batch_embed(self, content_items: List[Any]) -> Dict[int, np.ndarray]:
        """
        Create embeddings for multiple content items.
        
        Args:
            content_items: List of content items to embed
            
        Returns:
            Dictionary mapping indices to embedding vectors
        """
        pass
    
    @abstractmethod
    def can_handle(self, content: Any) -> bool:
        """
        Check if this handler can process the given content.
        
        Args:
            content: Content to check
            
        Returns:
            True if this handler can process the content, False otherwise
        """
        pass
    
    @abstractmethod
    def get_modality_type(self) -> ModalityType:
        """
        Get the modality type this handler supports.
        
        Returns:
            Modality type enum value
        """
        pass


class TextModalityHandler(ModalityHandler):
    """Handler for text content."""
    
    def __init__(self, config: ModalityConfig):
        """Initialize the text modality handler."""
        super().__init__(config)
        self.model_name = config.embedding_model
    
    def create_embedding(self, content: Any) -> Optional[np.ndarray]:
        """Create an embedding for text content."""
        if not self.can_handle(content):
            return None
            
        text = str(content)
        return get_embedding(text, model_name=self.model_name)
    
    def batch_embed(self, content_items: List[Any]) -> Dict[int, np.ndarray]:
        """Create embeddings for multiple text items."""
        result = {}
        valid_texts = []
        valid_indices = []
        
        # Filter and prepare valid texts
        for i, item in enumerate(content_items):
            if self.can_handle(item):
                valid_texts.append(str(item))
                valid_indices.append(i)
        
        # Batch embed all valid texts
        if valid_texts:
            embeddings = batch_embed(valid_texts, model_name=self.model_name)
            
            # Map back to original indices
            for i, text in enumerate(valid_texts):
                if text in embeddings:
                    result[valid_indices[i]] = embeddings[text]
        
        return result
    
    def can_handle(self, content: Any) -> bool:
        """Check if content is text or can be converted to text."""
        if content is None:
            return False
            
        # Content is already a string
        if isinstance(content, (str, bytes)):
            return True
            
        # Content has a string representation
        if hasattr(content, '__str__'):
            return True
            
        return False
    
    def get_modality_type(self) -> ModalityType:
        """Get the modality type."""
        return ModalityType.TEXT


class StructuredDataHandler(ModalityHandler):
    """Handler for structured data like JSON, dictionaries, or data frames."""
    
    def __init__(self, config: ModalityConfig):
        """Initialize the structured data handler."""
        super().__init__(config)
        self.model_name = config.embedding_model
        self.fields = config.metadata_fields or []
    
    def create_embedding(self, content: Any) -> Optional[np.ndarray]:
        """Create an embedding for structured data."""
        if not self.can_handle(content):
            return None
            
        # Extract text from structured data
        text = self._extract_text(content)
        if not text:
            return None
            
        return get_embedding(text, model_name=self.model_name)
    
    def batch_embed(self, content_items: List[Any]) -> Dict[int, np.ndarray]:
        """Create embeddings for multiple structured data items."""
        result = {}
        valid_texts = []
        valid_indices = []
        
        # Filter and prepare valid texts
        for i, item in enumerate(content_items):
            if self.can_handle(item):
                text = self._extract_text(item)
                if text:
                    valid_texts.append(text)
                    valid_indices.append(i)
        
        # Batch embed all valid texts
        if valid_texts:
            embeddings = batch_embed(valid_texts, model_name=self.model_name)
            
            # Map back to original indices
            for i, text in enumerate(valid_texts):
                if text in embeddings:
                    result[valid_indices[i]] = embeddings[text]
        
        return result
    
    def can_handle(self, content: Any) -> bool:
        """Check if content is structured data."""
        if content is None:
            return False
            
        # Dictionary or dict-like object
        if isinstance(content, dict):
            return True
            
        # JSON string
        if isinstance(content, str):
            try:
                json.loads(content)
                return True
            except json.JSONDecodeError:
                pass
        
        # Object with __dict__ attribute
        if hasattr(content, '__dict__'):
            return True
            
        return False
    
    def get_modality_type(self) -> ModalityType:
        """Get the modality type."""
        return ModalityType.STRUCTURED
    
    def _extract_text(self, content: Any) -> str:
        """Extract text from structured data."""
        if isinstance(content, str):
            try:
                # Try to parse as JSON
                content = json.loads(content)
            except json.JSONDecodeError:
                return content
        
        # Convert object to dictionary
        if not isinstance(content, dict) and hasattr(content, '__dict__'):
            content = content.__dict__
        
        # Extract specified fields or all fields
        if isinstance(content, dict):
            if self.fields:
                # Extract only specified fields
                values = []
                for field in self.fields:
                    if field in content:
                        values.append(str(content[field]))
                return " ".join(values)
            else:
                # Extract all fields
                return " ".join(str(v) for v in content.values())
        
        return str(content)


# Factory function to create appropriate modality handler
def create_modality_handler(config: ModalityConfig) -> Optional[ModalityHandler]:
    """
    Create a modality handler based on configuration.
    
    Args:
        config: Modality configuration
        
    Returns:
        Appropriate modality handler instance or None if not supported
    """
    if not config.enabled:
        return None
        
    if config.modality_type == ModalityType.TEXT:
        return TextModalityHandler(config)
    elif config.modality_type == ModalityType.STRUCTURED:
        return StructuredDataHandler(config)
    elif config.modality_type == ModalityType.CUSTOM and config.custom_handler:
        # Dynamic loading of custom handler
        try:
            module_path, class_name = config.custom_handler.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            handler_class = getattr(module, class_name)
            return handler_class(config)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load custom handler {config.custom_handler}: {str(e)}")
            return None
    else:
        logger.warning(f"Unsupported modality type: {config.modality_type}")
        return None


@register_plugin(
    plugin_type=PluginType.STORAGE,
    name="multi_modal_vector_store",
    version="1.0.0",
    description="Vector store supporting multiple content modalities",
    interfaces=["VectorStoreInterface"],
    is_default=False
)
class MultiModalVectorStore(VectorStoreInterface):
    """
    Vector store that supports multiple content modalities.
    
    This store maintains separate indices for different content types
    while providing a unified interface for storage and retrieval.
    """
    
    def __init__(
        self,
        persistence_path: Optional[str] = None,
        modality_configs: Optional[List[ModalityConfig]] = None,
        auto_save: bool = True,
        auto_save_threshold: int = 100
    ):
        """
        Initialize the multi-modal vector store.
        
        Args:
            persistence_path: Optional path for disk persistence
            modality_configs: List of modality configurations
            auto_save: Whether to automatically save on updates
            auto_save_threshold: Frequency of auto-saves (number of changes)
        """
        # Initialize modality configurations
        self.modality_configs = modality_configs or [
            # Default text modality config
            ModalityConfig(
                modality_type=ModalityType.TEXT,
                embedding_model=DEFAULT_MODEL,
                dimension=384,  # Default dimension for all-MiniLM-L6-v2
                enabled=True,
                similarity_threshold=0.6
            ),
            # Default structured data config
            ModalityConfig(
                modality_type=ModalityType.STRUCTURED,
                embedding_model=DEFAULT_MODEL,
                dimension=384,
                enabled=True,
                similarity_threshold=0.5
            )
        ]
        
        # Create modality handlers
        self.modality_handlers: Dict[ModalityType, ModalityHandler] = {}
        for config in self.modality_configs:
            handler = create_modality_handler(config)
            if handler:
                self.modality_handlers[config.modality_type] = handler
        
        # Storage for vectors and metadata by modality
        self.vectors: Dict[ModalityType, Dict[str, np.ndarray]] = {
            modality: {} for modality in self.modality_handlers.keys()
        }
        self.metadata: Dict[ModalityType, Dict[str, Dict[str, Any]]] = {
            modality: {} for modality in self.modality_handlers.keys()
        }
        
        # Cross-modal ID mapping
        self.id_to_modality: Dict[str, ModalityType] = {}
        
        # Track modifications
        self.modified: Set[str] = set()
        self.update_count: int = 0
        
        # Persistence settings
        self.persistence_path = persistence_path
        self.auto_save = auto_save
        self.auto_save_threshold = auto_save_threshold
        
        # Statistics
        self.stats = {
            "total_vectors": 0,
            "vectors_by_modality": {m.value: 0 for m in self.modality_handlers.keys()},
            "total_searches": 0,
            "created_at": time.time(),
            "last_modified": None
        }
        
        # Load from disk if persistence path provided
        if persistence_path:
            self._load()
        
        logger.info(f"Initialized MultiModalVectorStore with {len(self.modality_handlers)} modalities")
    
    def store(
        self,
        vectors: Dict[str, np.ndarray],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Store pre-computed vectors with optional metadata.
        
        Args:
            vectors: Dictionary mapping IDs to vector embeddings
            metadata: Optional dictionary mapping IDs to metadata dictionaries
            
        Returns:
            List of stored vector IDs
        """
        if not vectors:
            return []
            
        stored_ids = []
        metadata = metadata or {}
        
        for id, vector in vectors.items():
            # Determine modality
            modality = self._get_modality_from_metadata(id, metadata.get(id, {}))
            
            # Store vector and metadata
            self.vectors[modality][id] = vector
            
            # Store metadata
            if id in metadata:
                if id not in self.metadata[modality]:
                    self.metadata[modality][id] = {}
                self.metadata[modality][id].update(metadata[id])
            
            # Update modality mapping
            self.id_to_modality[id] = modality
            
            # Mark as modified
            self.modified.add(id)
            stored_ids.append(id)
        
        # Update stats
        self._update_stats()
        
        # Auto-save if enabled
        if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
            self._save()
            self.update_count = 0
        
        return stored_ids
    
    def store_content(
        self,
        content_items: Dict[str, Any],
        metadata: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[str]:
        """
        Store content items by auto-detecting modality and creating embeddings.
        
        Args:
            content_items: Dictionary mapping IDs to content items
            metadata: Optional dictionary mapping IDs to metadata dictionaries
            
        Returns:
            List of successfully stored IDs
        """
        if not content_items:
            return []
            
        stored_ids = []
        metadata = metadata or {}
        
        for id, content in content_items.items():
            # Determine modality and use appropriate handler
            handler = self._get_handler_for_content(content)
            if not handler:
                logger.warning(f"No suitable handler found for content with ID {id}")
                continue
                
            modality = handler.get_modality_type()
            
            # Create embedding
            vector = handler.create_embedding(content)
            if vector is None:
                logger.warning(f"Failed to create embedding for content with ID {id}")
                continue
            
            # Store vector
            self.vectors[modality][id] = vector
            
            # Store metadata
            if id in metadata:
                if id not in self.metadata[modality]:
                    self.metadata[modality][id] = {}
                self.metadata[modality][id].update(metadata[id])
            
            # Add content type to metadata if not present
            if id in self.metadata[modality]:
                if "content_type" not in self.metadata[modality][id]:
                    self.metadata[modality][id]["content_type"] = modality.value
            
            # Update modality mapping
            self.id_to_modality[id] = modality
            
            # Mark as modified
            self.modified.add(id)
            stored_ids.append(id)
        
        # Update stats
        self._update_stats()
        
        # Auto-save if enabled
        if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
            self._save()
            self.update_count = 0
        
        return stored_ids
    
    def retrieve(self, ids: List[str]) -> Dict[str, Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Retrieve vectors and metadata by IDs.
        
        Args:
            ids: List of vector IDs to retrieve
            
        Returns:
            Dictionary mapping IDs to tuples of (vector, metadata)
        """
        result = {}
        
        for id in ids:
            if id in self.id_to_modality:
                modality = self.id_to_modality[id]
                if id in self.vectors[modality]:
                    vector = self.vectors[modality][id]
                    meta = self.metadata[modality].get(id, {})
                    result[id] = (vector, meta)
        
        return result
    
    def search(
        self,
        query_vector: np.ndarray,
        limit: int = 10,
        threshold: float = 0.0,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        modalities: Optional[List[ModalityType]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for similar vectors across all modalities or specified ones.
        
        Args:
            query_vector: Vector to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            filter_fn: Optional function to filter results by metadata
            modalities: Optional list of modalities to search in (default: all)
            
        Returns:
            List of tuples containing (id, similarity, metadata)
        """
        # Track search count
        self.stats["total_searches"] += 1
        
        # Determine which modalities to search
        search_modalities = modalities or list(self.modality_handlers.keys())
        
        # Collect results from all specified modalities
        all_results = []
        
        for modality in search_modalities:
            if modality not in self.vectors:
                continue
                
            # Use modality-specific threshold if provided
            modality_threshold = threshold
            for config in self.modality_configs:
                if config.modality_type == modality:
                    modality_threshold = max(threshold, config.similarity_threshold)
                    break
            
            # Collect IDs to search
            ids_to_search = list(self.vectors[modality].keys())
            
            # Apply metadata filter if provided
            if filter_fn:
                ids_to_search = [
                    id for id in ids_to_search
                    if id in self.metadata[modality] and filter_fn(self.metadata[modality][id])
                ]
            
            # Calculate similarities
            for id in ids_to_search:
                vector = self.vectors[modality][id]
                
                # Calculate similarity
                similarity = cosine_similarity(query_vector, vector)
                
                # Apply threshold
                if similarity >= modality_threshold:
                    all_results.append(
                        (id, similarity, self.metadata[modality].get(id, {}))
                    )
        
        # Sort results by similarity (descending) and limit
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results[:limit]
    
    def search_by_content(
        self,
        query_content: Any,
        limit: int = 10,
        threshold: float = 0.0,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
        modalities: Optional[List[ModalityType]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search using content instead of a pre-computed vector.
        
        Args:
            query_content: Content to search for
            limit: Maximum number of results to return
            threshold: Minimum similarity threshold
            filter_fn: Optional function to filter results by metadata
            modalities: Optional list of modalities to search in (default: auto-detect)
            
        Returns:
            List of tuples containing (id, similarity, metadata)
        """
        # Determine modality and use appropriate handler
        handler = self._get_handler_for_content(query_content)
        if not handler:
            logger.warning(f"No suitable handler found for query content")
            return []
            
        # Create embedding
        query_vector = handler.create_embedding(query_content)
        if query_vector is None:
            logger.warning(f"Failed to create embedding for query content")
            return []
        
        # If no modalities specified, use the detected one and compatible ones
        if modalities is None:
            modalities = [handler.get_modality_type()]
        
        # Search with the computed vector
        return self.search(
            query_vector=query_vector,
            limit=limit,
            threshold=threshold,
            filter_fn=filter_fn,
            modalities=modalities
        )
    
    def delete(self, ids: List[str]) -> int:
        """
        Delete vectors by IDs.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            Number of vectors deleted
        """
        deleted_count = 0
        
        for id in ids:
            if id in self.id_to_modality:
                modality = self.id_to_modality[id]
                
                # Delete from vectors
                if id in self.vectors[modality]:
                    del self.vectors[modality][id]
                    deleted_count += 1
                
                # Delete from metadata
                if id in self.metadata[modality]:
                    del self.metadata[modality][id]
                
                # Delete from modality mapping
                del self.id_to_modality[id]
                
                # Remove from modified set
                if id in self.modified:
                    self.modified.remove(id)
        
        if deleted_count > 0:
            # Update stats
            self._update_stats()
            
            # Auto-save if enabled
            if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
                self._save()
                self.update_count = 0
        
        return deleted_count
    
    def update_metadata(self, id: str, metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a vector.
        
        Args:
            id: Vector ID to update
            metadata: New metadata to apply (merged with existing)
            
        Returns:
            True if successful, False otherwise
        """
        if id not in self.id_to_modality:
            return False
            
        modality = self.id_to_modality[id]
        
        # Initialize metadata dict if not exists
        if id not in self.metadata[modality]:
            self.metadata[modality][id] = {}
        
        # Update metadata
        self.metadata[modality][id].update(metadata)
        
        # Mark as modified
        self.modified.add(id)
        self.update_count += 1
        self.stats["last_modified"] = time.time()
        
        # Auto-save if enabled
        if self.auto_save and self.persistence_path and self.update_count >= self.auto_save_threshold:
            self._save()
            self.update_count = 0
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary of statistics
        """
        # Update vector counts
        self._update_stats()
        
        # Create result with current stats
        result = dict(self.stats)
        
        # Add modality information
        result["modalities"] = [
            {
                "type": config.modality_type.value,
                "model": config.embedding_model,
                "dimension": config.dimension,
                "enabled": config.enabled,
                "vector_count": self.stats["vectors_by_modality"].get(config.modality_type.value, 0)
            }
            for config in self.modality_configs
        ]
        
        # Memory usage (approximate)
        memory_usage = 0
        for modality, vectors in self.vectors.items():
            for vector in vectors.values():
                memory_usage += vector.nbytes
        
        result["memory_usage_bytes"] = memory_usage
        result["persistence_enabled"] = bool(self.persistence_path)
        result["auto_save_enabled"] = self.auto_save
        
        return result
    
    def clear(self) -> bool:
        """
        Clear all vectors from the store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear all data structures
            for modality in self.vectors.keys():
                self.vectors[modality].clear()
                self.metadata[modality].clear()
            
            self.id_to_modality.clear()
            self.modified.clear()
            
            # Update stats
            self._update_stats()
            self.update_count += 1
            
            # Auto-save if enabled
            if self.auto_save and self.persistence_path:
                self._save()
                self.update_count = 0
            
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_modality_configs(self) -> List[ModalityConfig]:
        """
        Get the current modality configurations.
        
        Returns:
            List of modality configurations
        """
        return self.modality_configs
    
    def add_modality(self, config: ModalityConfig) -> bool:
        """
        Add a new modality configuration.
        
        Args:
            config: Modality configuration to add
            
        Returns:
            True if successful, False otherwise
        """
        # Check if modality already exists
        for existing in self.modality_configs:
            if existing.modality_type == config.modality_type:
                logger.warning(f"Modality {config.modality_type} already exists")
                return False
        
        # Create handler
        handler = create_modality_handler(config)
        if not handler:
            logger.error(f"Failed to create handler for modality {config.modality_type}")
            return False
        
        # Add configuration and handler
        self.modality_configs.append(config)
        self.modality_handlers[config.modality_type] = handler
        
        # Initialize storage
        self.vectors[config.modality_type] = {}
        self.metadata[config.modality_type] = {}
        
        logger.info(f"Added modality {config.modality_type}")
        return True
    
    def update_modality(self, config: ModalityConfig) -> bool:
        """
        Update an existing modality configuration.
        
        Args:
            config: Updated modality configuration
            
        Returns:
            True if successful, False otherwise
        """
        # Find existing config
        for i, existing in enumerate(self.modality_configs):
            if existing.modality_type == config.modality_type:
                # Update configuration
                self.modality_configs[i] = config
                
                # Update handler if enabled
                if config.enabled:
                    handler = create_modality_handler(config)
                    if handler:
                        self.modality_handlers[config.modality_type] = handler
                else:
                    # Remove handler if disabled
                    if config.modality_type in self.modality_handlers:
                        del self.modality_handlers[config.modality_type]
                
                logger.info(f"Updated modality {config.modality_type}")
                return True
        
        logger.warning(f"Modality {config.modality_type} not found")
        return False
    
    def _update_stats(self) -> None:
        """Update storage statistics."""
        # Count vectors by modality
        for modality in self.vectors.keys():
            self.stats["vectors_by_modality"][modality.value] = len(self.vectors[modality])
        
        # Update total
        self.stats["total_vectors"] = sum(len(vectors) for vectors in self.vectors.values())
        
        # Update modification time
        self.stats["last_modified"] = time.time()
    
    def _get_modality_from_metadata(self, id: str, metadata: Dict[str, Any]) -> ModalityType:
        """Determine modality type from metadata."""
        # Check for explicit content_type in metadata
        if "content_type" in metadata and metadata["content_type"] in [m.value for m in ModalityType]:
            return ModalityType(metadata["content_type"])
            
        # Use modality from ID mapping if available
        if id in self.id_to_modality:
            return self.id_to_modality[id]
            
        # Default to TEXT modality
        return ModalityType.TEXT
    
    def _get_handler_for_content(self, content: Any) -> Optional[ModalityHandler]:
        """Find a suitable handler for the given content."""
        for handler in self.modality_handlers.values():
            if handler.can_handle(content):
                return handler
                
        return None
    
    def _save(self) -> bool:
        """Save to disk."""
        if not self.persistence_path:
            logger.warning("No persistence path specified, cannot save")
            return False
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.persistence_path), exist_ok=True)
            
            # Prepare data to save
            data = {
                "modality_configs": [config.to_dict() for config in self.modality_configs],
                "vectors": {modality.value: vectors for modality, vectors in self.vectors.items()},
                "metadata": {modality.value: metadata for modality, metadata in self.metadata.items()},
                "id_to_modality": {id: modality.value for id, modality in self.id_to_modality.items()},
                "stats": self.stats
            }
            
            # Save to disk
            with open(self.persistence_path, 'wb') as f:
                pickle.dump(data, f)
                
            logger.info(f"Saved {self.stats['total_vectors']} vectors to {self.persistence_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def _load(self) -> bool:
        """Load from disk."""
        if not self.persistence_path or not os.path.exists(self.persistence_path):
            return False
        
        try:
            with open(self.persistence_path, 'rb') as f:
                data = pickle.load(f)
            
            # Load modality configurations
            if "modality_configs" in data:
                self.modality_configs = [
                    ModalityConfig.from_dict(config_dict)
                    for config_dict in data["modality_configs"]
                ]
                
                # Recreate handlers
                self.modality_handlers = {}
                for config in self.modality_configs:
                    if config.enabled:
                        handler = create_modality_handler(config)
                        if handler:
                            self.modality_handlers[config.modality_type] = handler
            
            # Load vectors, converting string modality keys back to enum values
            if "vectors" in data:
                self.vectors = {
                    ModalityType(modality_str): vectors
                    for modality_str, vectors in data["vectors"].items()
                    if ModalityType(modality_str) in self.modality_handlers
                }
            
            # Load metadata
            if "metadata" in data:
                self.metadata = {
                    ModalityType(modality_str): metadata
                    for modality_str, metadata in data["metadata"].items()
                    if ModalityType(modality_str) in self.modality_handlers
                }
            
            # Load ID to modality mapping
            if "id_to_modality" in data:
                self.id_to_modality = {
                    id: ModalityType(modality_str)
                    for id, modality_str in data["id_to_modality"].items()
                }
            
            # Load stats
            if "stats" in data:
                self.stats = data["stats"]
            
            # Initialize missing data structures
            for modality in self.modality_handlers.keys():
                if modality not in self.vectors:
                    self.vectors[modality] = {}
                if modality not in self.metadata:
                    self.metadata[modality] = {}
            
            logger.info(f"Loaded {self.stats.get('total_vectors', 0)} vectors from {self.persistence_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return False