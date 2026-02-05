"""
Machine Translation Evaluation System

To run this project:
1. Install dependencies: pip install -r requirements.txt
2. (Optional) Create a .env file with GOOGLE_TRANSLATE_API_KEY and GEMINI_API_KEY
3. Run the server: uvicorn app.main:app --reload

This architecture uses FastAPI for its high performance and async capabilities.
The services are separated to ensure modularity and ease of testing/swapping engines.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.utils.logger import logger

from app.api.back_translate import router as back_translate_router
from app.services.nllb_translate import nllb_service
from app.services.evaluation import evaluation_service
from app.core.config import settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    import os
    import torch
    
    # Configure PyTorch memory management to reduce fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Set HF_TOKEN environment variable if provided
    if settings.HF_TOKEN:
        os.environ["HF_TOKEN"] = settings.HF_TOKEN
        
    # Clear CUDA cache at startup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    # Startup: Pre-load models
    print("-" * 50)
    print(f"🚀 Starting {settings.APP_NAME}")
    print(f"📌 Gemini Model: {settings.GEMINI_MODEL_NAME}")
    print(f"📌 NLLB Model: {settings.NLLB_MODEL_NAME}")
    print(f"📌 Max Concurrent Rows: {settings.CONCURRENT_ROWS}")
    print(f"📌 Max Concurrent Eval Tasks: {settings.CONCURRENT_EVAL_TASKS}")
    print(f"🛡️ API Protection: Gemini={settings.GEMINI_MAX_CONCURRENT}/s, Google={settings.GOOGLE_MAX_CONCURRENT}/s")
    print("-" * 50)
    
    logger.info("Starting up and pre-loading models...")
    nllb_service.load_model() 
    evaluation_service.load_model()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    # Shutdown logic if any
    logger.info("Shutting down...")

app = FastAPI(
    title=settings.APP_NAME,
    description="Evaluate machine translation quality using back-translation.",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
# app.include_router(translate_router, tags=["Translation"]) # Disabled as requested
app.include_router(back_translate_router, tags=["Back Translation"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Machine Translation Evaluation System", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
