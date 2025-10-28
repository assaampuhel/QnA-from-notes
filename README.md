# 🧠 QnA from Notes — Level 3 Mini RAG System

## 🎯 Objective
This project is a **Retrieval-Augmented Generation (RAG) System** that allows you to ask questions directly from your notes.  
It retrieves the most relevant sections from your notes and generates concise, human-like answers using a local Transformer model.

---

## 🚀 Features
- 📂 Upload your own **text notes** (`.txt`)
- 🔍 Retrieve **most relevant sections** using **TF-IDF + cosine similarity**
- 🧠 Generate **summarized answers** using **Hugging Face Transformers**
- 💬 Automatic **extractive fallback** when no transformer is available
- ⚡ Caching for **fast, repeated generation**
- 🧾 Fully **offline capable**
- 🧹 Automatic **NLTK stopword download and fallback**

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Streamlit** — for the user interface  
- **scikit-learn** — for vectorization and similarity search  
- **NLTK** — for text preprocessing  
- **Transformers (Hugging Face)** — for text generation  
- **PyTorch** — as the backend for the transformer model

---

## 📦 Installation

### Step 1: Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # On macOS/Linux
venv\Scripts\activate         # On Windows

### Step 2: Install dependencies
```bash
pip install streamlit nltk torch scikit-learn transformers sentencepiece

### Step 3: Download NLTK stopwords
```bash
python
>>> import nltk
>>> nltk.download('stopwords')
>>> exit()

---

## ▶️ Usage
```bash
streamlit run app.py

---

## ⚙️ How It Works

1. **Text Processing:**  
   Splits your notes into paragraphs and cleans up the text.

2. **Retrieval:**  
   Uses **TF-IDF + cosine similarity** to find the most relevant paragraphs for your question.

3. **Generation:**  
   - If **enabled**, uses a **transformer summarization model** like `t5-small` or `facebook/bart-large-cnn` to generate an answer.  
   - If **disabled or offline**, falls back to a **fast extractive summarizer** that selects the most relevant sentences.

---

## 📦 Folder Structure

QnA-from-Notes/
│
├── app.py               # Main Streamlit app
├── sample_notes.txt     # Example notes file
├── README.md            # Documentation (this file)
└── venv/                # Optional virtual environment folder

---

## 🧩 Common Issues & Solutions

| **Issue** | **Solution** |
|------------|---------------|
| `nltk_data not found` | Run `import nltk; nltk.download('stopwords')` manually in Python. |
| Transformer download slow | Use a smaller model (`t5-small`) or enable **extractive mode** instead of transformer mode. |
| Torch errors on install | Use: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| App not starting | Ensure you’re running inside the **virtual environment** (`venv`). |

---

## 💡 Future Enhancements
Add PDF and DOCX support
Introduce multi-document search
Add semantic memory for multi-turn Q&A
Integrate with LLMs like LLaMA or Mistral locally
Option to export Q&A sessions to a .txt or .csv file

Feel free to fork this repo, make improvements, and submit pull requests!
You can also:
    Try other summarization models (like google/pegasus-xsum)
    Replace TF-IDF with semantic embeddings (e.g., sentence-transformers)

## 📜 License

This project is released under the MIT License.
You are free to use, modify, and distribute it with attribution.

