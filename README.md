# OCR

A modular Streamlit app that:
1. Accepts an image.
2. Extracts English text using PaddleOCR.
3. Converts extracted text into dense vector embeddings using `sentence-transformers/all-MiniLM-L6-v2`.
4. Stores the text, vector, and metadata in a local persistent ChromaDB collection.
5. Retrieves similar stored text using semantic similarity.

## Folder structure

```text
ocr_chromadb_project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── utils/
│   ├── __init__.py
│   ├── image_utils.py
│   └── ocr_engine.py
├── services/
│   ├── __init__.py
│   ├── embedding_service.py
│   └── chroma_service.py
└── data/
    └── chroma_db/       # created automatically; do not commit this folder
```

## Setup on Windows PowerShell

Use a supported Python version for your PaddlePaddle build (Python 3.10 or 3.11 is a practical choice for many environments). Activate your virtual environment first.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install PaddlePaddle using the installation command recommended for your operating system, Python version, and CPU/GPU setup from the official PaddlePaddle installation guide. Then install the remaining requirements:

```powershell
python -m pip install -r requirements.txt
```

If the generic `paddlepaddle` install fails, install a compatible PaddlePaddle wheel first using the official instructions, then rerun requirements installation.

## Run

```powershell
python -m streamlit run app.py --server.port 8502
```

Open `http://localhost:8502`.

## First run and model downloads

The first OCR operation may download PaddleOCR model files. The first embedding operation may download `sentence-transformers/all-MiniLM-L6-v2` from the Hugging Face model hub. Internet access is required unless the models have already been cached locally.

## Data flow

```text
Image
  -> Pillow validation / RGB conversion
  -> PaddleOCR detection + recognition
  -> extracted text
  -> Sentence Transformers embedding vector
  -> ChromaDB persistent collection
  -> semantic query embedding
  -> nearest matching stored text
```

## Notes

- This project uses pretrained models; it does not train OCR or embedding models.
- Embeddings are numeric representations of text meaning, not the original text itself. ChromaDB stores both the vector and the document text.
- Search is semantic similarity, not a guarantee of factual correctness.
- The default ChromaDB data directory is `data/chroma_db/`.
- The app allows editing extracted text before storing it.
- Keep sensitive documents local and do not publish the `data/chroma_db/` directory.
