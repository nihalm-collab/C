import os
from dotenv import load_dotenv
import streamlit as st
from datasets import load_dataset
from haystack.document_stores import ChromaDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptNode
from haystack.pipelines import GenerativeQAPipeline
from sentence_transformers import SentenceTransformer

# 1️⃣ .env yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# 2️⃣ Streamlit arayüz başlığı
st.title("📚 Kitap Yorumları Chatbotu")
st.markdown("""
Bu chatbot, Kitapyurdu yorum verisi üzerinde çalışır ve sorularınızı yorumlardan bulup Gemini API ile cevap üretir.
""")

# 3️⃣ Veri seti yükleme (HF token ile)
@st.cache_data
def load_kitapyurdu_data():
    dataset = load_dataset(
        "alibayram/kitapyurdu_yorumlar",
        split="train",
        use_auth_token=HF_TOKEN  # HF token burada kullanılıyor
    )
    documents = []
    for item in dataset:
        documents.append({
            "content": item["review_text"],
            "meta": {"product": item.get("product_title", "")}
        })
    return documents

documents = load_kitapyurdu_data()

# 4️⃣ ChromaDB DocumentStore kur
doc_store = ChromaDocumentStore(
    persist_directory="./chroma_db",
    embedding_dim=384
)

# Eğer DB boşsa doküman ekle
if len(doc_store.get_all_documents()) == 0:
    doc_store.write_documents(documents)

# 5️⃣ Retriever kur (embedding modeli)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
retriever = EmbeddingRetriever(
    document_store=doc_store,
    embedding_model=embedding_model,
    use_gpu=False
)

# 6️⃣ Generator / LLM Node (Gemini)
generator = PromptNode(
    model_name_or_path="google/gemini-2.0",
    api_key=GEMINI_API_KEY,
    default_prompt_template="""
You are an expert assistant answering questions about books based on user reviews.
Use retrieved context to answer precisely and concisely.

Context:
{documents}

Question:
{query}
"""
)

# 7️⃣ RAG Pipeline
pipe = GenerativeQAPipeline(generator=generator, retriever=retriever)

# 8️⃣ Kullanıcı input
user_input = st.text_input("Sorunuzu yazın:")

if user_input:
    result = pipe.run(query=user_input, params={"Retriever": {"top_k": 3}})
    answer = result["answers"][0].answer
    st.subheader("Cevap:")
    st.write(answer)
