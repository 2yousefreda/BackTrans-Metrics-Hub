import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from app.core.config import settings
import asyncio
from concurrent.futures import ThreadPoolExecutor

class NLLBTranslateService:
    _instance = None
    _model = None
    _tokenizer = None
    _executor = ThreadPoolExecutor(max_workers=1)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NLLBTranslateService, cls).__new__(cls)
        return cls._instance

    def load_model(self):
        """Loads the model and tokenizer (Singleton)"""
        if self._model is None:
            print(f"Loading NLLB model: {settings.NLLB_MODEL_NAME}...")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.NLLB_MODEL_NAME,
                src_lang="eng_Latn"
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                settings.NLLB_MODEL_NAME,
                torch_dtype=torch.float32
            )
            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self.device)
            print(f"✅ NLLB Model loaded on {self.device}")

    def _sync_translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """NLLB translation with proper FLORES-200 language code handling"""
        # FLORES-200 language codes for NLLB
        lang_map = {
            "en": "eng_Latn",
            "ar": "arb_Arab",  # IMPORTANT: Use arb_Arab (Modern Standard Arabic), NOT ara_Arab 
            "fr": "fra_Latn",
            "es": "spa_Latn"
        }
        
        src_code = lang_map.get(source_lang, source_lang)
        tgt_code = lang_map.get(target_lang, target_lang)

        # Set source language on tokenizer
        self._tokenizer.src_lang = src_code
        
        # Tokenize input
        inputs = self._tokenizer(text, return_tensors="pt").to(self.device)
        
        # Get the token ID for the target language
        # The target language token must be used as the forced_bos_token_id
        tgt_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        
        print(f"🔄 NLLB: {src_code} → {tgt_code} (Token ID: {tgt_token_id})")
        
        # Generate translation
        with torch.inference_mode():
            generated_tokens = self._model.generate(
                **inputs,
                forced_bos_token_id=tgt_token_id,  # Force the decoder to start with target language token
                max_new_tokens=100,  # Use max_new_tokens instead of max_length for better control
                num_beams=1  # Greedy decoding for speed
            )
        
        # Decode the output
        translated_text = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        
        print(f"✅ Translation: '{text}' → '{translated_text}'")
        
        return translated_text.strip()

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text using local NLLB-200 model.
        Wrapper to run the sync model inference in a thread pool.
        """
        if self._model is None:
            self.load_model()
            
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, 
            self._sync_translate, 
            text, 
            source_lang, 
            target_lang
        )

# Global instance
nllb_service = NLLBTranslateService()
