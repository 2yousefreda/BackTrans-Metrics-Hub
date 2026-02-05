"""
Evaluation Service for Translation Quality Assessment
Provides three evaluation metrics:
1. Cosine Similarity using GATE-AraBert-v1 embeddings
2. BERTScore
3. BLEU Score
"""

import numpy as np
from typing import Dict, Optional
import threading
from app.core.config import settings

class EvaluationService:
    """Service for evaluating translation quality"""
    
    def __init__(self):
        self._embedding_model = None
        self._bert_scorer = None
        self._eval_lock = threading.Semaphore(settings.CONCURRENT_EVAL_TASKS)
    
    @property
    def embedding_model(self):
        """Lazy load or return the pre-loaded GATE-AraBert model"""
        with self._eval_lock:
            if self._embedding_model is None:
                self.load_model()
            return self._embedding_model
    
    def load_model(self):
        """Load GATE-AraBert-v1 model into memory"""
        from sentence_transformers import SentenceTransformer
        import torch
        if self._embedding_model is None:
            print("Starting up and pre-loading GATE-AraBert-v1 model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer(
                'Omartificial-Intelligence-Space/GATE-AraBert-v1',
                device=device
            )
            print(f"✅ GATE-AraBert-v1 model loaded successfully on {device}!")
    
    def cosine_similarity(self, original: str, back_translated: str) -> float:
        """
        Calculate cosine similarity between original and back-translated text
        """
        with self._eval_lock:
            try:
                if not original or not back_translated:
                    return 0.0
                
                # Get embeddings
                embeddings = self._embedding_model.encode([original, back_translated])
                
                # Calculate cosine similarity
                vec1, vec2 = embeddings[0], embeddings[1]
                similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                return float(similarity)
            except Exception as e:
                print(f"Cosine similarity error: {e}")
                return 0.0
    
    def bert_score(self, original: str, back_translated: str) -> float:
        """
        Calculate BERTScore between original and back-translated text.
        """
        with self._eval_lock:
            try:
                if not original or not back_translated:
                    return 0.0
                
                from bert_score import score
                
                # Calculate BERTScore
                P, R, F1 = score(
                    [back_translated], 
                    [original], 
                    lang="ar",
                    verbose=False
                )
                
                return float(F1[0])
            except Exception as e:
                print(f"BERTScore error: {e}")
                if "CUDA out of memory" in str(e):
                    import torch
                    torch.cuda.empty_cache()
                return 0.0
    
    def bleu_score(self, original: str, back_translated: str) -> float:
        """
        Calculate BLEU score between original and back-translated text.
        
        Args:
            original: Original Arabic text (reference)
            back_translated: Back-translated Arabic text (hypothesis)
            
        Returns:
            BLEU score (0 to 100, higher is better)
        """
        try:
            if not original or not back_translated:
                return 0.0
            
            from sacrebleu.metrics import BLEU
            
            bleu = BLEU()
            result = bleu.sentence_score(back_translated, [original])
            
            return float(result.score)
        except Exception as e:
            print(f"BLEU score error: {e}")
            return 0.0
    
    def evaluate_all(self, original: str, back_translated: str) -> Dict[str, float]:
        """
        Run all evaluation metrics.
        
        Args:
            original: Original Arabic text
            back_translated: Back-translated Arabic text
            
        Returns:
            Dictionary with all scores
        """
        return {
            'cosine_similarity': self.cosine_similarity(original, back_translated),
            'bert_score': self.bert_score(original, back_translated),
            'bleu_score': self.bleu_score(original, back_translated)
        }


# Singleton instance
evaluation_service = EvaluationService()
