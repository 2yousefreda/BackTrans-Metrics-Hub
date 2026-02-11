
import asyncio
import sys
import os

# Add the project root to sys.path
sys.path.append(os.getcwd())

from app.services.evaluation import evaluation_service

async def test_eval():
    original = "تعتبر اللغة العربية من أقدم اللغات. وهي لغة القرآن الكريم."
    # Back-translated with one sentence slightly changed and one significantly changed
    back_translated = "تعتبر اللغة العربية من أقدم اللغات. وهي لغة الكتاب الكريم."
    
    print("Testing Evaluation Service...")
    print(f"Original: {original}")
    print(f"Back-translated: {back_translated}")
    
    # Pre-load model manually to avoid concurrent loading issues in test
    evaluation_service.load_model()
    
    results = evaluation_service.evaluate_all(original, back_translated)
    
    import json
    print("\nEvaluation Results:")
    print(json.dumps(results, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(test_eval())
