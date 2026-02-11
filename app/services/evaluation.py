"""
Evaluation Service for Translation Quality Assessment
Provides three evaluation metrics:
1. Cosine Similarity using GATE-AraBert-v1 embeddings
2. BERTScore
3. BLEU Score
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import threading
import re
from app.core.config import settings

# Thresholds for decision logic
DEFAULT_THRESHOLDS = {
    "mbert":   {"bs_accept": 0.88, "bs_reject": 0.80},
    "arabert": {"bs_accept": 0.82, "bs_reject": 0.72},
    "xlmr":    {"bs_accept": 0.78, "bs_reject": 0.65},

    # cosine thresholds for GATE-AraBERT sentence embeddings
    "cos_sent_accept_mean": 0.80,
    "cos_sent_reject_min":  0.55,

    # strict “min sentence” gates (very effective vs hallucinations) 
    "bs_min_accept": 0.65,
    "bs_min_reject": 0.45,
}

def _resolve_bertscore_model(model_key: str) -> str:
    key = (model_key or "").lower().strip()
    if key in ("xlmr", "xlm-r", "xlm-roberta-large", "xlmroberta-large"):
        return "xlm-roberta-large"
    if key in ("arabert", "arabertv2", "aubmindlab/bert-base-arabertv2"):
        return "aubmindlab/bert-base-arabertv2"
    if key in ("mbart", "mbert", "bert-base-multilingual-cased", "multilingual-bert"):
        return "bert-base-multilingual-cased"
    # User preferred xlm-roberta-large or arabertv2 for Arabic
    return "xlm-roberta-large"

class EvaluationService:
    """Service for evaluating translation quality"""
    
    def __init__(self):
        self._embedding_model = None
        self._bert_scorer = None
        self._sentence_splitter = None
    
    @property
    def embedding_model(self):
        """Lazy load or return the pre-loaded GATE-AraBert model"""
        if self._embedding_model is None:
            self.load_model()
        return self._embedding_model
    
    def load_model(self):
        """Load GATE-AraBert-v1 model into memory"""
        from sentence_transformers import SentenceTransformer
        import torch
        if self._embedding_model is None:
            print("Starting up and pre-loading GATE-AraBert-v1 model...")
            from sentence_transformers import SentenceTransformer
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._embedding_model = SentenceTransformer(
                'Omartificial-Intelligence-Space/GATE-AraBert-v1',
                device=device
            )
            print(f"✅ GATE-AraBert-v1 model loaded successfully on {device}!")
            
            # Pre-warm BERTScore model (xlm-roberta-large)
            print("Pre-warming BERTScore model (xlm-roberta-large)...")
            from bert_score import score as bertscore_fn
            with torch.inference_mode():
                # Just a tiny call to initialize the model in memory
                bertscore_fn(["test"], ["test"], model_type="xlm-roberta-large", lang="ar")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("✅ BERTScore model (XLM-R Large) pre-warmed!")
    
    def split_sentences(self, text: str) -> List[str]:
        """Split Arabic text into sentences using CAMeL Tools or regex fallback"""
        if not text:
            return []
            
        try:
            from camel_tools.tokenizers.word import simple_word_tokenize
            # Try multiple possible paths for SentenceSplitter
            try:
                from camel_tools.tokenizers.sentence import SentenceSplitter
            except ImportError:
                # Fallback to regex if sentence splitter is not in the expected path
                raise ImportError("SentenceSplitter not found in camel_tools.tokenizers.sentence")
            
            if self._sentence_splitter is None:
                self._sentence_splitter = SentenceSplitter()
            
            # Use simple_word_tokenize function
            tokens = simple_word_tokenize(text)
            sentences = self._sentence_splitter.split(tokens)
            return sentences
        except Exception:
            # Simple regex-based splitter as fallback
            AR_SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\؟\?؛])\s+|[\n\r]+")
            sents = [s.strip() for s in AR_SENT_SPLIT_RE.split(text) if s and s.strip()]
            sentences = [s for s in sents if len(s) >= 2]
            return sentences

    def cosine_similarity(self, original: str, back_translated: str) -> float:
        """
        Calculate cosine similarity between original and back-translated text
        """
        try:
            if not original or not back_translated:
                return 0.0
            
            # Get embeddings
            embeddings = self.embedding_model.encode([original, back_translated])
                
            # Calculate cosine similarity
            vec1, vec2 = embeddings[0], embeddings[1]
            denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            if denom == 0:
                return 0.0
            similarity = np.dot(vec1, vec2) / denom
            
            return float(similarity)
        except Exception as e:
            print(f"Cosine similarity error: {e}")
            return 0.0

    def sentence_cosine_stats(self, original: str, back_translated: str) -> Dict[str, Union[float, int]]:
        """
        Sentence-level cosine similarity statistics
        """
        if not original or not back_translated:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        s1 = self.split_sentences(original)
        s2 = self.split_sentences(back_translated)
        n = min(len(s1), len(s2))
        if n == 0:
            return {"mean": 0.0, "min": 0.0, "max": 0.0, "count": 0}

        s1, s2 = s1[:n], s2[:n]
        # Use lazy-loaded embedding model
        emb = self.embedding_model.encode(s1 + s2)
        a = emb[:n]
        b = emb[n:]

        sims = []
        for i in range(n):
            u, v = a[i], b[i]
            denom = (np.linalg.norm(u) * np.linalg.norm(v))
            sims.append(float(np.dot(u, v) / denom) if denom != 0 else 0.0)

        res = {
            "mean": float(sum(sims) / n),
            "min": float(min(sims)),
            "max": float(max(sims)),
            "count": int(n),
        }
        print(f"    - Cosine (Sentence-Level): Mean={res['mean']:.4f}, Min={res['min']:.4f}")
        return res
    
    def bert_score(
        self,
        original: str,
        back_translated: str,
        model_key: str = "arabert",
    ) -> float:
        """Calculate document-level BERTScore"""
        model_type = _resolve_bertscore_model(model_key)
        
        try:
            if not original or not back_translated:
                return 0.0
            
            from bert_score import score as bertscore_fn
            print(f"    - Running BERTScore (Doc) using {model_type}...")
            
            try:
                # Try with rescaling first
                P, R, F1 = bertscore_fn(
                    [back_translated], 
                    [original], 
                    model_type=model_type,
                    lang="ar",
                    verbose=False,
                    rescale_with_baseline=True
                )
            except Exception as e:
                print(f"    - BERTScore baseline fallback: {e}")
                # Fallback to without rescaling
                P, R, F1 = bertscore_fn(
                    [back_translated], 
                    [original], 
                    model_type=model_type,
                    lang="ar",
                    verbose=False,
                    rescale_with_baseline=False
                )
            
            score_val = float(F1[0])
            print(f"    - BERTScore (Doc) Match: {score_val:.4f}")
            return score_val
        except Exception as e:
            print(f"BERTScore error: {e}")
            if "CUDA out of memory" in str(e):
                import torch
                torch.cuda.empty_cache()
            return 0.0

    def bert_score_min_sentence(
        self,
        original: str,
        back_translated: str,
        model_key: str = "arabert",
    ) -> Dict[str, Union[float, int]]:
        """Sentence-level BERTScore (Min/Mean)"""
        if not original or not back_translated:
            return {"mean": 0.0, "min": 0.0, "count": 0}

        s1 = self.split_sentences(original)
        s2 = self.split_sentences(back_translated)
        n = min(len(s1), len(s2))
        if n == 0:
            return {"mean": 0.0, "min": 0.0, "count": 0}

        s1, s2 = s1[:n], s2[:n]

        try:
            from bert_score import score as bertscore_fn
            model_type = _resolve_bertscore_model(model_key)
            print(f"    - Running BERTScore (Sentences) using {model_type}...")

            try:
                # Try with rescaling first
                P, R, F1 = bertscore_fn(
                    s2,
                    s1,
                    model_type=model_type,
                    lang="ar",
                    verbose=False,
                    rescale_with_baseline=True,
                )
            except Exception as e:
                print(f"    - BERTScore baseline fallback (sent): {e}")
                # Fallback to no rescaling
                P, R, F1 = bertscore_fn(
                    s2,
                    s1,
                    model_type=model_type,
                    lang="ar",
                    verbose=False,
                    rescale_with_baseline=False,
                )
            
            f1s = [float(x) for x in F1]
            res_dict = {
                "mean": float(sum(f1s) / n),
                "min": float(min(f1s)),
                "count": int(n),
            }
            print(f"    - BERTScore (Sent-Level): Mean={res_dict['mean']:.4f}, Min={res_dict['min']:.4f}")
            return res_dict
        except Exception as e:
            print(f"BERTScore sentence error: {e}")
            return {"mean": 0.0, "min": 0.0, "count": int(n)}
    
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
    
    def auto_decision(
        self,
        cos_sent: Dict[str, Union[float, int]],
        bs_doc: float,
        bs_sent: Dict[str, Union[float, int]],
        model_key: str,
        thresholds: Optional[Dict] = None,
    ) -> Dict[str, object]:
        """Final decision based on quality gates"""
        t = thresholds or DEFAULT_THRESHOLDS
        mk = (model_key or "").lower()
        
        if "xlm" in mk:
            m = t["xlmr"]
        elif "mbert" in mk or "multilingual" in mk:
            m = t["mbert"]
        else:
            m = t["arabert"]

        # Hard reject gates (catch catastrophic failures)
        if cos_sent["count"] == 0 or bs_sent["count"] == 0:
            return {"decision": "REVIEW", "reason": "empty_sentence_split"}

        if cos_sent["min"] < t["cos_sent_reject_min"]:
            return {"decision": "REJECT", "reason": "low_min_sentence_cosine"}

        if bs_sent["min"] < t["bs_min_reject"]:
            return {"decision": "REJECT", "reason": "low_min_sentence_bertscore"}

        if bs_doc < m["bs_reject"]:
            return {"decision": "REJECT", "reason": "low_document_bertscore"}

        # Accept gates
        if (cos_sent["mean"] >= t["cos_sent_accept_mean"] and
            bs_doc >= m["bs_accept"] and
            bs_sent["min"] >= t["bs_min_accept"]):
            return {"decision": "ACCEPT", "reason": "passes_all_accept_gates"}

        # Otherwise review
        result = {"decision": "REVIEW", "reason": "borderline_metrics"}
        print(f"    - Final Decision: {result['decision']} ({result['reason']})")
        return result

    def evaluate_all(self, original: str, back_translated: str, model_key: str = "arabert") -> Dict[str, object]:
        """
        Run all evaluation metrics and generate a decision.
        """
        cos_sent = self.sentence_cosine_stats(original, back_translated)
        bs_doc = self.bert_score(original, back_translated, model_key)
        bs_sent = self.bert_score_min_sentence(original, back_translated, model_key)
        bleu = self.bleu_score(original, back_translated)
        
        decision = self.auto_decision(cos_sent, bs_doc, bs_sent, model_key)
        
        return {
            'cosine_doc': self.cosine_similarity(original, back_translated),
            'cosine_sent': cos_sent,
            'bert_score_doc': bs_doc,
            'bert_score_sent': bs_sent,
            'bleu': bleu,
            'decision': decision
        }


# Singleton instance
evaluation_service = EvaluationService()
