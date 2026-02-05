"""
Back-Translation API Endpoint
Accepts an Excel file and performs back-translation using selected engines.
Returns the results as an Excel file with optional evaluation scores.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
import pandas as pd
import asyncio
import time
from io import BytesIO
from typing import Optional

from app.services.gemini_translate import gemini_service
from app.services.nllb_translate import nllb_service
from app.services.google_translate import google_service
from app.services.evaluation import evaluation_service
from app.core.config import settings
from app.utils.logger import logger

router = APIRouter()

# Language detection mapping
LANG_MAP = {
    'en': ['english', 'en', 'eng', 'إنجليزي'],
    'fr': ['french', 'fr', 'français', 'فرنسي'],
    'es': ['spanish', 'es', 'español', 'إسباني'],
    'de': ['german', 'de', 'deutsch', 'ألماني']
}


def detect_language(header: str) -> str:
    """Detect language from column header"""
    header_lower = header.lower()
    for lang_code, patterns in LANG_MAP.items():
        if any(pattern in header_lower for pattern in patterns):
            return lang_code
    return 'en'  # Default to English


async def back_translate_row(text: str, source_lang: str, use_gemini: bool, use_nllb: bool, use_google: bool) -> dict:
    """
    Translate a single text using selected methods back to Arabic in parallel
    """
    tasks = []
    engine_names = []
    
    if use_gemini:
        tasks.append(gemini_service.translate(text, source_lang, 'ar'))
        engine_names.append('gemini')
    
    if use_nllb:
        tasks.append(nllb_service.translate(text, source_lang, 'ar'))
        engine_names.append('nllb')
        
    if use_google:
        tasks.append(google_service.translate(text, source_lang, 'ar'))
        engine_names.append('google')
        
    if not tasks:
        return {}

    # Run translations in parallel
    results_list = await asyncio.gather(*tasks, return_exceptions=True)
    
    results = {}
    for name, res in zip(engine_names, results_list):
        if isinstance(res, Exception):
            results[name] = f"ERROR: {str(res)}"
        else:
            results[name] = res
            
    return results


async def evaluate_translation_async(original: str, back_translated: str, enable_evaluation: bool) -> dict:
    """
    Evaluate translation quality asynchronously (using threads for CPU bound tasks)
    """
    if not enable_evaluation or not back_translated or back_translated.startswith("ERROR:"):
        return {
            'cosine_sim': None,
            'bert_score': None,
            'bleu': None
        }
    
    # Run heavy CPU-bound evaluation in a thread pool
    scores = await asyncio.to_thread(evaluation_service.evaluate_all, original, back_translated)
    
    return {
        'cosine_sim': round(scores['cosine_similarity'], 4),
        'bert_score': round(scores['bert_score'], 4),
        'bleu': round(scores['bleu_score'], 2)
    }


@router.post("/back-translate")
async def back_translate_excel(
    file: UploadFile = File(..., description="Excel file with Arabic text and translations"),
    use_gemini: bool = Form(True, description="Use Gemini for translation"),
    use_nllb: bool = Form(True, description="Use NLLB for translation"),
    use_google: bool = Form(True, description="Use Google Translate"),
    enable_evaluation: bool = Form(True, description="Enable quality evaluation (Cosine Similarity, BERTScore, BLEU)")
):
    """
    Back-Translation Endpoint
    
    Accepts an Excel file with:
    - Column 1: Arabic text
    - Column 2: Translation (header specifies target language)
    
    Engine Selection (all enabled by default):
    - use_gemini: Enable/disable Gemini translation
    - use_nllb: Enable/disable NLLB translation  
    - use_google: Enable/disable Google Translate
    
    Evaluation (enabled by default):
    - enable_evaluation: Calculate quality scores for each translation
    
    Returns an Excel file with columns for each selected engine and evaluation scores.
    """
    
    # Validate at least one engine is selected
    if not any([use_gemini, use_nllb, use_google]):
        raise HTTPException(
            status_code=400,
            detail="At least one translation engine must be selected"
        )
    
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload an Excel file (.xlsx or .xls)"
        )
    
    try:
        # Start timer
        start_time = time.time()
        
        # Read the uploaded file
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        
        if len(df) > settings.MAX_ROWS_PER_FILE:
            raise HTTPException(
                status_code=400,
                detail=f"File exceeds maximum allowed rows ({settings.MAX_ROWS_PER_FILE})"
            )

        if len(df.columns) < 2:
            raise HTTPException(
                status_code=400,
                detail="Excel file must have at least 2 columns (Arabic and Translation)"
            )
        
        # Get column names
        col_arabic = df.columns[0]
        col_translation = df.columns[1]
        
        # Detect source language from column header
        source_lang = detect_language(col_translation)
        
        # Prepare result columns based on selected engines
        results_data = {
            'gemini': [] if use_gemini else None,
            'nllb': [] if use_nllb else None,
            'google': [] if use_google else None
        }
        
        # Prepare evaluation columns
        eval_data = {
            'gemini': {'cosine': [], 'bert': [], 'bleu': []} if use_gemini else None,
            'nllb': {'cosine': [], 'bert': [], 'bleu': []} if use_nllb else None,
            'google': {'cosine': [], 'bert': [], 'bleu': []} if use_google else None
        }
        
        # Use a Semaphore to limit concurrent rows from config
        semaphore = asyncio.Semaphore(settings.CONCURRENT_ROWS)
        
        async def process_row_task(idx, row):
            async with semaphore:
                text = str(row[col_translation])
                original_arabic = str(row[col_arabic])
                
                row_result = {
                    'idx': idx,
                    'gemini': "", 'nllb': "", 'google': "",
                    'gemini_scores': {'cosine': None, 'bert': None, 'bleu': None},
                    'nllb_scores': {'cosine': None, 'bert': None, 'bleu': None},
                    'google_scores': {'cosine': None, 'bert': None, 'bleu': None}
                }
                
                if pd.isna(text) or text.strip() == '' or text == 'nan':
                    return row_result
                
                # 1. Back-Translate
                translations = await back_translate_row(text, source_lang, use_gemini, use_nllb, use_google)
                
                # 2. Evaluate each engine's result in parallel
                eval_tasks = []
                engines_to_eval = []
                
                for engine in ['gemini', 'nllb', 'google']:
                    if translations.get(engine) and not translations[engine].startswith("ERROR:"):
                        row_result[engine] = translations[engine]
                        if enable_evaluation:
                            eval_tasks.append(evaluate_translation_async(original_arabic, translations[engine], enable_evaluation))
                            engines_to_eval.append(engine)
                    elif translations.get(engine):
                        row_result[engine] = translations[engine]
                
                if eval_tasks:
                    eval_results = await asyncio.gather(*eval_tasks)
                    for engine, scores in zip(engines_to_eval, eval_results):
                        row_result[f"{engine}_scores"] = {
                            'cosine': scores['cosine_sim'],
                            'bert': scores['bert_score'],
                            'bleu': scores['bleu']
                        }
                
                return row_result

        # Create tasks for all rows with a tiny delay to avoid burst rate limiting
        tasks = []
        for idx, row in df.iterrows():
            tasks.append(process_row_task(idx, row))
            await asyncio.sleep(0.05) # Jitter/Burst protection
        
        # Run all rows concurrently with semaphore protection
        all_row_results = await asyncio.gather(*tasks)
        
        # Sort results by index to maintain order
        all_row_results.sort(key=lambda x: x['idx'])
        
        # Populate the dataframe
        if use_gemini:
            df['Gemini_BackTranslation'] = [r['gemini'] for r in all_row_results]
            if enable_evaluation:
                df['Gemini_CosineSim'] = [r['gemini_scores']['cosine'] for r in all_row_results]
                df['Gemini_BERTScore'] = [r['gemini_scores']['bert'] for r in all_row_results]
                df['Gemini_BLEU'] = [r['gemini_scores']['bleu'] for r in all_row_results]
        
        if use_nllb:
            df['NLLB_BackTranslation'] = [r['nllb'] for r in all_row_results]
            if enable_evaluation:
                df['NLLB_CosineSim'] = [r['nllb_scores']['cosine'] for r in all_row_results]
                df['NLLB_BERTScore'] = [r['nllb_scores']['bert'] for r in all_row_results]
                df['NLLB_BLEU'] = [r['nllb_scores']['bleu'] for r in all_row_results]
        
        if use_google:
            df['Google_BackTranslation'] = [r['google'] for r in all_row_results]
            if enable_evaluation:
                df['Google_CosineSim'] = [r['google_scores']['cosine'] for r in all_row_results]
                df['Google_BERTScore'] = [r['google_scores']['bert'] for r in all_row_results]
                df['Google_BLEU'] = [r['google_scores']['bleu'] for r in all_row_results]
        
        # End timer
        duration = time.time() - start_time
        logger.info(f"⏱️ Total processing time for {len(df)} rows: {duration:.2f} seconds")
        logger.info(f"📊 Average speed: {duration/len(df):.2f} seconds per row")
        
        # Create output Excel file in memory
        output = BytesIO()
        df.to_excel(output, index=False, engine='openpyxl')
        output.seek(0)
        
        # Generate output filename
        original_name = file.filename.rsplit('.', 1)[0]
        output_filename = f"{original_name}_back_translated.xlsx"
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={output_filename}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

