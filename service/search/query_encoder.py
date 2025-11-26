"""
Query Encoder for Smart Search.

Encode user queries into semantic embeddings using PhoBERT.
Supports Vietnamese text with abbreviation expansion and caching.

Example:
    >>> from service.search.query_encoder import QueryEncoder
    >>> encoder = QueryEncoder()
    >>> emb = encoder.encode("kem dưỡng da cho da dầu")
    >>> embeddings = encoder.encode_batch(["query1", "query2"])
"""

from typing import List, Optional, Dict, Tuple
from pathlib import Path
from collections import OrderedDict
import numpy as np
import logging
import threading
import re
import time

logger = logging.getLogger(__name__)


# ============================================================================
# Vietnamese Abbreviations & Preprocessing
# ============================================================================

VIETNAMESE_ABBREVIATIONS = {
    # Product abbreviations
    'sp': 'sản phẩm',
    'kem dc': 'kem dưỡng da',
    'kem dd': 'kem dưỡng da',
    'srm': 'sữa rửa mặt',
    'tbc': 'tẩy bong chết',
    'tdc': 'tẩy da chết',
    'kcn': 'kem chống nắng',
    'cn': 'chống nắng',
    'nc': 'nước',
    'nct': 'nước cân bằng',
    'ncb': 'nước cân bằng',
    'nht': 'nước hoa hồng',
    'nhh': 'nước hoa hồng',
    'ta': 'toner',
    'nc ht': 'nước hoa hồng',
    'dd': 'dưỡng da',
    'da': 'da',
    'mt': 'mặt',
    'mat': 'mặt',
    
    # Skin type abbreviations
    'dn': 'da nhờn',
    'dk': 'da khô',
    'dh': 'da hỗn hợp',
    'dhh': 'da hỗn hợp',
    'dnc': 'da nhạy cảm',
    'dm': 'da mụn',
    
    # Common adjectives
    'bt': 'bình thường',
    'hq': 'hiệu quả',
    'ok': 'tốt',
    'ko': 'không',
    'k': 'không',
    'dc': 'được',
    'đc': 'được',
    'j': 'gì',
    'z': 'vậy',
    'v': 'vậy',
    'vs': 'với',
    'r': 'rồi',
    'lm': 'làm',
    'cx': 'cũng',
    'nt': 'như thế',
    'ns': 'nói',
    'ntn': 'như thế nào',
    'bn': 'bao nhiêu',
    'bnh': 'bao nhiêu',
}

# Positive/negative keywords for potential query understanding
POSITIVE_KEYWORDS = [
    'tốt', 'hiệu quả', 'thấm nhanh', 'mịn', 'sáng', 'đẹp', 
    'thơm', 'dễ chịu', 'dịu nhẹ', 'an toàn', 'phù hợp'
]

NEGATIVE_KEYWORDS = [
    'không tốt', 'kém', 'dở', 'gây mụn', 'kích ứng', 
    'nhờn', 'bết', 'khô', 'nứt', 'bong tróc'
]


class LRUCache:
    """Simple LRU cache for query embeddings."""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[np.ndarray]:
        """Get item from cache, moving to end if found."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None
    
    def put(self, key: str, value: np.ndarray) -> None:
        """Put item in cache, evicting oldest if at capacity."""
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    self.cache.popitem(last=False)
                self.cache[key] = value
    
    def clear(self) -> None:
        """Clear all cached items."""
        with self._lock:
            self.cache.clear()
    
    def __len__(self) -> int:
        return len(self.cache)
    
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        return {
            'size': len(self.cache),
            'capacity': self.capacity
        }


class QueryEncoder:
    """
    Encode text queries to embeddings using PhoBERT.
    
    Features:
    - Lazy loading of PhoBERT model
    - Query embedding caching (LRU)
    - Batch encoding for efficiency
    - Vietnamese text preprocessing with abbreviation expansion
    
    Example:
        >>> encoder = QueryEncoder()
        >>> emb = encoder.encode("kem dưỡng da cho da dầu")
        >>> embeddings = encoder.encode_batch(["query1", "query2"])
    """
    
    _instance: Optional['QueryEncoder'] = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern for resource efficiency."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(
        self,
        model_name: str = "AITeamVN/Vietnamese_Embedding",
        max_length: int = 256,
        cache_size: int = 1000,
        device: str = "cpu",
        abbreviations: Optional[Dict[str, str]] = None
    ):
        """
        Initialize QueryEncoder.
        
        Args:
            model_name: HuggingFace model name for PhoBERT
            max_length: Maximum sequence length for tokenization
            cache_size: Size of LRU cache for query embeddings
            device: Device for model inference ('cpu' or 'cuda')
            abbreviations: Additional abbreviation mappings
        """
        if self._initialized:
            return
        
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        
        # Abbreviations
        self.abbreviations = VIETNAMESE_ABBREVIATIONS.copy()
        if abbreviations:
            self.abbreviations.update(abbreviations)
        
        # Build regex pattern for abbreviation replacement
        # Sort by length (longest first) to avoid partial matches
        sorted_abbrs = sorted(self.abbreviations.keys(), key=len, reverse=True)
        escaped_abbrs = [re.escape(abbr) for abbr in sorted_abbrs]
        self._abbr_pattern = re.compile(
            r'\b(' + '|'.join(escaped_abbrs) + r')\b',
            re.IGNORECASE
        )
        
        # Lazy loaded model components
        self._tokenizer = None
        self._model = None
        self._model_loaded = False
        
        # Query cache
        self._cache = LRUCache(capacity=cache_size)
        
        # Statistics
        self._stats = {
            'queries_encoded': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_encoding_time_ms': 0.0
        }
        
        self._initialized = True
        logger.info(f"QueryEncoder initialized (model: {model_name}, cache: {cache_size})")
    
    def _load_model(self) -> None:
        """Lazy load PhoBERT model and tokenizer."""
        if self._model_loaded:
            return
        
        start = time.perf_counter()
        
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Loading PhoBERT model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
            
            elapsed = (time.perf_counter() - start) * 1000
            logger.info(f"PhoBERT model loaded in {elapsed:.1f}ms on {self.device}")
            
            self._model_loaded = True
            
        except ImportError as e:
            logger.error(f"Failed to import transformers/torch: {e}")
            raise ImportError(
                "QueryEncoder requires 'transformers' and 'torch' packages. "
                "Install with: pip install transformers torch"
            )
        except Exception as e:
            logger.error(f"Failed to load PhoBERT model: {e}")
            raise
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess Vietnamese query text.
        
        Steps:
        1. Lowercase and strip whitespace
        2. Expand abbreviations
        3. Normalize whitespace
        4. Remove special characters (keep Vietnamese)
        
        Args:
            query: Raw query text
        
        Returns:
            Preprocessed query string
        """
        # Lowercase and strip
        text = query.lower().strip()
        
        # Expand abbreviations
        def replace_abbr(match):
            abbr = match.group(1).lower()
            return self.abbreviations.get(abbr, abbr)
        
        text = self._abbr_pattern.sub(replace_abbr, text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Vietnamese diacritics
        # Keep: letters, numbers, spaces, Vietnamese characters (using Unicode ranges)
        # Vietnamese Unicode ranges: \u00C0-\u024F (Latin Extended), \u1E00-\u1EFF (Latin Extended Additional)
        text = re.sub(r'[^\w\s\u00C0-\u024F\u1E00-\u1EFF]', '', text, flags=re.UNICODE)
        
        return text.strip()
    
    def encode(
        self,
        query: str,
        normalize: bool = True,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Encode a single query to embedding.
        
        Args:
            query: Text query (Vietnamese)
            normalize: L2 normalize the embedding for cosine similarity
            use_cache: Use LRU cache for repeated queries
        
        Returns:
            np.ndarray of shape (embedding_dim,)
        """
        start = time.perf_counter()
        
        # Preprocess
        processed_query = self.preprocess_query(query)
        
        # Cache key
        cache_key = f"{processed_query}_{normalize}"
        
        # Check cache
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                self._stats['cache_hits'] += 1
                return cached
            self._stats['cache_misses'] += 1
        
        # Load model if needed
        self._load_model()
        
        import torch
        
        # Tokenize
        inputs = self._tokenizer(
            processed_query,
            max_length=self.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Use [CLS] token embedding (first token)
            # Handle different model output formats
            if hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
            elif hasattr(outputs, 'pooler_output'):
                embedding = outputs.pooler_output.cpu().numpy()[0]
            else:
                # Fallback: use first element if it's a tuple
                embedding = outputs[0][:, 0, :].cpu().numpy()[0]
        
        # Normalize for cosine similarity
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        # Cache result
        if use_cache:
            self._cache.put(cache_key, embedding)
        
        # Update stats
        elapsed = (time.perf_counter() - start) * 1000
        self._stats['queries_encoded'] += 1
        self._stats['total_encoding_time_ms'] += elapsed
        
        return embedding
    
    def encode_batch(
        self,
        queries: List[str],
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode multiple queries efficiently with batching.
        
        Args:
            queries: List of text queries
            normalize: L2 normalize embeddings
            batch_size: Batch size for encoding
            show_progress: Show progress bar (requires tqdm)
        
        Returns:
            np.ndarray of shape (num_queries, embedding_dim)
        """
        if not queries:
            return np.array([])
        
        self._load_model()
        
        import torch
        
        all_embeddings = []
        
        # Setup progress bar if requested
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(0, len(queries), batch_size), desc="Encoding queries")
            except ImportError:
                iterator = range(0, len(queries), batch_size)
        else:
            iterator = range(0, len(queries), batch_size)
        
        for i in iterator:
            batch = queries[i:i + batch_size]
            
            # Preprocess all queries in batch
            processed = [self.preprocess_query(q) for q in batch]
            
            # Tokenize batch
            inputs = self._tokenizer(
                processed,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode batch
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Handle different model output formats
                if hasattr(outputs, 'last_hidden_state'):
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                elif hasattr(outputs, 'pooler_output'):
                    batch_embeddings = outputs.pooler_output.cpu().numpy()
                else:
                    # Fallback: use first element if it's a tuple
                    batch_embeddings = outputs[0][:, 0, :].cpu().numpy()
            
            all_embeddings.append(batch_embeddings)
        
        # Concatenate all batches
        result = np.vstack(all_embeddings)
        
        # Normalize if requested
        if normalize:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
            result = result / norms
        
        self._stats['queries_encoded'] += len(queries)
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the query embedding cache."""
        self._cache.clear()
        logger.info("Query encoder cache cleared")
    
    def get_stats(self) -> Dict[str, any]:
        """Get encoder statistics."""
        stats = self._stats.copy()
        stats['cache'] = self._cache.stats()
        stats['model_loaded'] = self._model_loaded
        
        # Compute average encoding time
        if stats['queries_encoded'] > 0:
            stats['avg_encoding_time_ms'] = (
                stats['total_encoding_time_ms'] / stats['queries_encoded']
            )
        else:
            stats['avg_encoding_time_ms'] = 0.0
        
        # Cache hit rate
        total_cache_access = stats['cache_hits'] + stats['cache_misses']
        if total_cache_access > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_cache_access
        else:
            stats['cache_hit_rate'] = 0.0
        
        return stats
    
    @property
    def embedding_dim(self) -> int:
        """Get embedding dimension."""
        if self._model is not None:
            return self._model.config.hidden_size
        # Default for PhoBERT-base
        return 768
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded


# ============================================================================
# Singleton Access Functions
# ============================================================================

def get_query_encoder(**kwargs) -> QueryEncoder:
    """
    Get singleton QueryEncoder instance.
    
    Args:
        **kwargs: Arguments passed to QueryEncoder constructor
    
    Returns:
        QueryEncoder instance
    """
    return QueryEncoder(**kwargs)


def reset_query_encoder() -> None:
    """Reset singleton QueryEncoder instance (for testing)."""
    with QueryEncoder._lock:
        if QueryEncoder._instance is not None:
            QueryEncoder._instance._cache.clear()
        QueryEncoder._instance = None
    logger.info("QueryEncoder singleton reset")
