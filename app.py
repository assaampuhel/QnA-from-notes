"""
QnA-from-Notes — Final Level 3 (Best Version)
- Retrieval: TF-IDF + cosine similarity
- Generation: summarization pipeline (default: t5-small) OR extractive fallback
- Robust: NLTK auto-download fallback, model caching, UI options
- Usage: streamlit run app.py
"""

import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# NLTK stopwords (auto-download if needed)
# --------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        # network/ssl might fail; we'll fallback to a small builtin set below
        pass

try:
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    STOPWORDS = set([
        "the", "is", "in", "and", "to", "of", "a", "that", "it", "as", "for", "with", "on", "was", "are",
        "this", "these", "those", "be", "by", "an", "or", "from", "at", "which"
    ])

# --------------------------
# Text utilities
# --------------------------
def clean_text(text: str) -> str:
    """Lowercase, remove extra whitespace (keep punctuation for sentence splitting)."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()

def split_into_paragraphs(text: str):
    """Split text into paragraphs using blank lines or double newlines."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras

def preprocess_for_tfidf(text: str) -> str:
    """Lowercase and remove punctuation for TF-IDF preprocessing (stopwords removal done by vectorizer)."""
    t = text.lower()
    t = re.sub(f"[{re.escape(string.punctuation)}]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

# --------------------------
# Retrieval: TF-IDF + Cosine similarity
# --------------------------
def retrieve_top_k(question: str, paragraphs: list, top_k: int = 3):
    """Return list of tuples (index, paragraph, score) sorted by relevance descending."""
    if not paragraphs:
        return []
    # Build TF-IDF on paragraphs + question
    vectorizer = TfidfVectorizer(stop_words='english')
    corpus = paragraphs + [question]
    tfidf = vectorizer.fit_transform(corpus)
    q_vec = tfidf[-1]
    para_vecs = tfidf[:-1]
    sims = cosine_similarity(q_vec, para_vecs).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [(int(i), paragraphs[int(i)], float(sims[int(i)])) for i in top_indices]

# --------------------------
# Extractive fallback summarizer (fast, deterministic)
# --------------------------
def extractive_summarize(question: str, context: str, max_sentences: int = 3) -> str:
    """
    Simple extractive summarizer:
    - Split context into sentences.
    - Score sentences by token overlap with the question.
    - Return top sentences (original order).
    """
    # Basic sentence splitter (keeps punctuation)
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if s.strip()]
    if not sentences:
        return ""
    q_tokens = set(preprocess_for_tfidf(question).split()) - STOPWORDS
    scored = []
    for idx, s in enumerate(sentences):
        s_tokens = set(preprocess_for_tfidf(s).split()) - STOPWORDS
        overlap = len(q_tokens & s_tokens)
        scored.append((idx, s, overlap))
    # sort by score desc, then take top N, then restore original order
    top = sorted(scored, key=lambda x: (-x[2], x[0]))[:max_sentences]
    top_sorted_by_idx = sorted(top, key=lambda x: x[0])
    answer = " ".join([s for (_, s, _) in top_sorted_by_idx])
    # fallback: if overlap zero for all, return first N sentences
    if not answer.strip():
        answer = " ".join(sentences[:max_sentences])
    return answer

# --------------------------
# Transformer summarizer (cached)
# --------------------------
@st.cache_resource(show_spinner=False)
def load_summarizer(model_name: str):
    """
    Load a Hugging Face summarization pipeline for the chosen model.
    Default recommended: 't5-small' (fast & small) or 'facebook/bart-large-cnn' (higher quality).
    """
    from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
    # Ensure small models finish faster on CPU — pipeline will download and cache
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
    return summarizer

def transformer_summarize(question: str, context: str, summarizer, max_length=120, min_length=30):
    """
    Build a prompt-like input for summarization models: include question and context.
    Many seq2seq summarizers accept a single text; we concatenate question & context.
    """
    # Keep the context length reasonable — truncate if huge
    max_context_tokens = 800  # rough limit to avoid OOM on CPU
    # simple truncation by characters (not tokens) to be safe
    if len(context) > 2000:
        context = context[:2000] + " ..."
    input_text = "question: " + question.strip() + "\ncontext: " + context.strip()
    # call the summarizer pipeline
    out = summarizer(input_text, max_length=max_length, min_length=min_length, do_sample=False)
    if out and isinstance(out, list):
        return out[0].get('summary_text', '').strip()
    return ""

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="QnA-from-Notes — Mini RAG (Final)", layout="wide")
st.title("📚 QnA-from-Notes — Mini RAG (Final)")
st.markdown("""
This app retrieves relevant paragraphs from your notes using **TF-IDF** and generates a concise answer using a **summarization model** (or an extractive fallback).
- Recommended model for speed: **t5-small**
- Higher-quality option: **facebook/bart-large-cnn** (larger)
""")

# File upload
uploaded_file = st.file_uploader("Upload notes (.txt)", type=["txt"])
model_choice = st.selectbox("Summarization model (smaller = faster):", options=["t5-small", "facebook/bart-large-cnn"])
use_transformer = st.checkbox("Use transformer summarizer (may be slower on CPU)", value=True)
top_k = st.slider("How many paragraphs to retrieve for context", min_value=1, max_value=5, value=3)

if uploaded_file:
    raw_text = clean_text(uploaded_file.read().decode("utf-8"))
    paragraphs = split_into_paragraphs(raw_text)
    st.success(f"Loaded {len(paragraphs)} paragraph(s).")

    question = st.text_input("Ask a question about your notes:")
    if st.button("Get Answer"):
        if not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Retrieving relevant paragraphs..."):
                retrieved = retrieve_top_k(question, paragraphs, top_k=top_k)

            if not retrieved:
                st.error("No relevant paragraphs found in notes.")
            else:
                st.subheader("📚 Retrieved Evidence")
                for rank, (idx, para, score) in enumerate(retrieved, start=1):
                    st.markdown(f"**Rank {rank} — Paragraph #{idx+1} — Similarity: {score:.3f}**")
                    st.write(para)

                # build context from retrieved paragraphs (join)
                context = " ".join([para for (_, para, _) in retrieved])

                # Generate answer
                st.subheader("🧠 Generated Answer")
                answer = ""

                if use_transformer:
                    try:
                        with st.spinner(f"Loading '{model_choice}' summarizer (cached)..."):
                            summarizer = load_summarizer(model_choice)
                        with st.spinner("Generating summarized answer from context..."):
                            answer = transformer_summarize(question, context, summarizer)
                            if not answer:
                                raise ValueError("Transformer returned empty summary — falling back.")
                    except Exception as e:
                        st.warning(f"Transformer summarizer failed or is unavailable: {e}\nFalling back to extractive summarizer.")
                        answer = extractive_summarize(question, context)
                else:
                    # purely extractive
                    answer = extractive_summarize(question, context)

                st.success(answer)

                st.markdown("---")
                st.subheader("🔎 Full Context Used (for transparency)")
                st.code(context, language="text")

else:
    st.info("Upload a .txt file (your notes) to begin.")