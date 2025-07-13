# ========================
# [COMMON SETUP]
# ========================
import os
import tempfile
from pathlib import Path
import gradio as gr
from dotenv import load_dotenv

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import APIClient

from langchain_community.document_loaders import PyPDFLoader  # Task 1
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Task 2
from langchain_community.embeddings import HuggingFaceEmbeddings  # Task 3
from langchain_community.vectorstores import Chroma  # Task 4
from langchain.chains import RetrievalQA  # Task 6
from langchain.llms.base import LLM  # Task 6

# Load environment variables
env_path = Path(__file__).resolve().parent / ".env"
print(f"[DEBUG] Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path)

# Debug output
api_key = os.getenv("WATSONX_API_KEY") or ""
project_id = os.getenv("WATSONX_PROJECT_ID") or ""
raw_url = os.getenv("WATSONX_URL") or ""
model_id = os.getenv("MODEL_ID") or ""

url = raw_url.strip()
if url and not url.startswith("https://"):
    url = "https://" + url

if not all([api_key, project_id, url, model_id]):
    raise EnvironmentError("❌ Missing Watsonx credentials")


# Setup credentials
wml_credentials: dict[str, str] = {
    "apikey": api_key,
    "url": url
}
client = APIClient(wml_credentials=wml_credentials)

# ========================
# TASK 1: PDF Loader
# ========================
def load_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    print(f"[TASK 1] Loaded {len(documents)} pages")
    return documents


# ========================
# TASK 2: Text Splitter
# ========================
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"[TASK 2] Split into {len(chunks)} chunks")
    return chunks


# ========================
# TASK 3: Embedding
# ========================
def embed_chunks(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("[TASK 3] Embedding complete")
    return embedder, chunks


# ========================
# TASK 4: Vector DB
# ========================
def build_vectordb(chunks, embedder):
    vectordb = Chroma.from_documents(chunks, embedder)
    print("[TASK 4] Chroma vector DB created")
    return vectordb


# ========================
# TASK 5: Retriever
# ========================
def build_retriever(vectordb):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    print("[TASK 5] Retriever ready")
    return retriever


# ========================
# Watsonx LLM Class (Used in Task 6)
# ========================
class WatsonxLLM(LLM):
    def __init__(self):
        super().__init__()
        self._model = ModelInference(
            model_id=model_id,
            api_client=client,
            project_id=project_id,
            params={
                "decoding_method": "greedy",
                "max_new_tokens": 300,
                "temperature": 0.5
            }
        )

    def _call(self, prompt: str, stop=None):
        print(f"[PROMPT] {prompt}")
        response = self._model.generate(prompt)
        print(f"[RESPONSE] {response}")
        if isinstance(response, dict) and "results" in response:
            return response["results"][0]["generated_text"]
        elif isinstance(response, list) and len(response) > 0 and "generated_text" in response[0]:
            return response[0]["generated_text"]
        else:
            return "[ERROR] Unexpected LLM response format"

    @property
    def _llm_type(self):
        return "watsonx"

# ========================
# TASK 6: QA Bot + Gradio UI
# ========================
qa_chain_global = {"chain": None}

def process_pdf(file_path):
    docs = load_pdf(file_path)  # Task 1
    chunks = split_docs(docs)  # Task 2
    embedder, chunks = embed_chunks(chunks)  # Task 3
    vectordb = build_vectordb(chunks, embedder)  # Task 4
    retriever = build_retriever(vectordb)  # Task 5
    llm = WatsonxLLM()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    print("[TASK 6] QA Bot Ready")
    return qa_chain


def upload_pdf_and_process(file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file)
            tmp_path = tmp.name
        qa_chain = process_pdf(tmp_path)
        qa_chain_global["chain"] = qa_chain
        return "✅ File processed. Ask your question."
    except Exception as e:
        return f"[ERROR] Upload failed: {e}"


def query_llm(input_query):
    qa_chain = qa_chain_global.get("chain")
    if not qa_chain:
        return "⚠️ Upload a PDF first."
    return qa_chain.run(input_query)


def launch_gradio_ui():
    with gr.Blocks(title="Watsonx RAG Assistant") as demo:
        gr.Markdown("## 🧠DocuReadAI")
        with gr.Row():
            file_input = gr.File(label="Upload PDF", type="binary")
            upload_button = gr.Button("Process")
        status = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            question = gr.Textbox(label="Your Question")
            ask_button = gr.Button("Ask")
        answer = gr.Textbox(label="Answer", interactive=False)

        upload_button.click(fn=upload_pdf_and_process, inputs=[file_input], outputs=[status])
        ask_button.click(fn=query_llm, inputs=[question], outputs=[answer])

    demo.launch()


# ========== ENTRY POINT ==========
if __name__ == "__main__":
    print("[SYSTEM] Launching Watsonx RAG Assistant...")
    launch_gradio_ui()
