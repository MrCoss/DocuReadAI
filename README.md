# DocuReadAI

A **Document Understanding Assistant** powered by LangChain, HuggingFace Embeddings, Chroma VectorDB, and IBM Watsonx LLMs. Built as part of the Coursera capstone project: _"Generative AI Applications with RAG and LangChain"_.

---

## 🔍 Overview

**DocuReadAI** enables real-time document analysis and question answering. It reads PDFs, splits content intelligently, embeds it into a vector store, and answers queries using IBM Watsonx's large language models.

Built with:
- 🧠 **LangChain** for pipeline logic
- 📚 **HuggingFace Embeddings** for document vectorization
- 💾 **Chroma** as a local vector database
- ☁️ **Watsonx.ai** for LLM inference
- 🎛️ **Gradio** for a simple user interface

---

## 🚀 Features

- Upload and process any PDF document
- Query content via natural language
- Retrieve exact answers from document context
- Powered by Watsonx foundation models

---

## 🏗️ Architecture

```text
PDF → LangChain Splitter → HuggingFace Embeddings → ChromaDB → Retriever → Watsonx LLM → Answer
🛠️ Setup Instructions
1. Clone the repository
git clone https://github.com/MrCoss/DocuReadAI.git
cd DocuReadAI

2. Create .env file
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://eu-gb.ml.cloud.ibm.com
MODEL_ID=google/flan-ul2

3. Install dependencies
pip install -r requirements.txt

4. Run the assistant
python main.py
📷 Screenshots
Required for Coursera grading — include these in a /screenshots folder:
pdf_loader.png – PDF loading code
code_splitter.png – Text splitting logic
embedding.png – HuggingFace embedding code
vectordb.png – Chroma DB configuration
retriever.png – Retriever setup

QA_bot.png – Gradio UI with sample query

📁 File Structure
bash
Copy
Edit
.
├── main.py                # Core app logic
├── .env                  # Your Watsonx API credentials (not committed)
├── requirements.txt      # Python dependencies
├── .gitignore            # Ignored files and folders
└── README.md             # Project documentation

📄 License
This project is released under the MIT License.

👤 Author
Costas Pinto
GitHub: MrCoss
Coursera Capstone - IBM / DeepLearning.AI
