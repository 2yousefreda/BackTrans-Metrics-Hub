# BackTrans-Metrics-Hub

A high-performance FastAPI-based system designed to evaluate the quality of machine translations through **Back-Translation** and advanced linguistic metrics.

## 🚀 Features

- **Multi-Engine Translation**: Parallel back-translation using:
  - **Google Gemini 3 Flash** (Latest high-speed LLM)
  - **Google Translate API** (Commercial NMT)
  - **Meta NLLB-200** (Open-source SOTA model)
- **Quality Metrics**: Automated scoring comparing original vs. back-translated text:
  - **Cosine Similarity**: Semantic similarity using `GATE-AraBert-v1`.
  - **BERTScore**: Leveraging context-aware embeddings for Arabic.
  - **SacreBLEU**: Traditional n-gram overlap metric.
- **Production-Ready Performance**:
  - **Global Rate Limit Protection**: Prevents API bans using centralized semaphores for Gemini and Google Translate.
  - **Resilient Retry System**: Automatic exponential backoff for network or quota errors.
  - **Multi-Threading**: Evaluation tasks and NLLB inference run in optimized thread pools.
  - **Full Async Design**: Non-blocking I/O for high throughput.
- **Excel Processing**: Batch processing with column auto-detection and safety row limits.

## 📋 Prerequisites

- **Python**: 3.10 or higher.
- **Hardware**: NVIDIA GPU (6GB+ VRAM recommended) with CUDA support for best performance.
- **API Keys**:
  - Google Cloud Console (for Google Translate API).
  - Google AI Studio (for Gemini API).
  - Hugging Face Token (for NLLB model access).

## 🛠️ Installation

### 1. Automatic Setup (Windows - Recommended)

The fastest way to set up the project is to run the `setup_project.bat` script. It will automatically create the virtual environment and install all dependencies:

```bash
./setup_project.bat
```

### 2. Manual Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/2yousefreda/BackTrans-Metrics-Hub.git
   cd BackTrans-Metrics-Hub
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory (refer to `.env.example`):

   ```env
   # API Keys
   GEMINI_API_KEY=your_gemini_key
   GOOGLE_TRANSLATE_API_KEY=your_google_key
   HF_TOKEN=your_huggingface_token

   # AI Version
   GEMINI_MODEL_NAME=gemini-3-flash-preview

   # Rate Limit & Concurrency
   CONCURRENT_ROWS=5
   GEMINI_MAX_CONCURRENT=2
   GOOGLE_MAX_CONCURRENT=2
   CONCURRENT_EVAL_TASKS=2

   # Safety
   MAX_ROWS_PER_FILE=1000
   ```

## 🖥️ Usage

1. **Start the Server**

   ```bash
   venv\Scripts\uvicorn app.main:app --reload
   ```

2. **Access API Documentation**
   Open your browser and navigate to:
   - Swagger UI: `http://localhost:8000/docs`
   - Redoc: `http://localhost:8000/redoc`

3. **Batch Processing**
   Use the `/back-translate` endpoint to upload an Excel file.
   - **Column 1**: Original Arabic text.
   - **Column 2**: Translated text (e.g., English, French).
   - The system will detect the language, perform back-translation to Arabic, and append similarity scores.

## Excel File Structure

The system expects an Excel file (`.xlsx` or `.xls`) with at least two columns.

### Input Format

| Column A (Index 0)  | Column B (Index 1)                                                 |
| :------------------ | :----------------------------------------------------------------- |
| **Original Arabic** | **Target Translation**                                             |
| (Must be Arabic)    | (Header must contain language name like "English", "French", etc.) |

**Example:**
| Arabic Source | English Translation |
| :--- | :--- |
| مرحباً بك في نظامنا | Welcome to our system |

### Output Format

The system will return a new Excel file containing all original columns plus:

- **Engine Back-Translation**: One column for each engine (Gemini, NLLB, Google).
- **Quality Scores**: Three columns for each engine (Cosine Similarity, BERTScore, BLEU).

## 📁 Project Structure

```text
├── app/
│   ├── api/          # API Route definitions (/back-translate)
│   ├── core/         # Global configuration & settings
│   ├── services/     # Translation engine & evaluation logic
│   ├── utils/        # Shared utilities (Logger, etc.)
│   └── main.py       # Application entry point & Middlewares
├── .env              # Sensitive credentials (DO NOT COMMIT)
├── .env.example      # Example environment variables
├── requirements.txt  # Python dependencies
└── setup_project.bat # Automatic environment setup script
```

## ⚖️ License

This project is for research and evaluation purposes. Use in accordance with the licenses of the respective translation engines (Google, Meta).
