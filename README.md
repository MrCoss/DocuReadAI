# DocuReadAI: An Intelligent Document Q\&A Assistant

[](https://shields.io/)
[](https://www.python.org/)
[](https://www.langchain.com/)
[](https://www.ibm.com/watsonx)

**DocuReadAI** is an intelligent document understanding assistant built using **LangChain**, **HuggingFace**, **Chroma DB**, and **IBM Watsonx**. It provides a seamless interface to "chat" with your PDF documents, allowing you to extract specific insights through natural language queries.
-----

## Table of Contents

  - [1. Project Vision & Problem Statement](https://www.google.com/search?q=%231-project-vision--problem-statement)
  - [2. The RAG Pipeline Explained: How It Works](https://www.google.com/search?q=%232-the-rag-pipeline-explained-how-it-works)
  - [3. Core Features & Functionality](https://www.google.com/search?q=%233-core-features--functionality)
  - [4. Technical Stack](https://www.google.com/search?q=%234-technical-stack)
  - [5. Project Structure Explained](https://www.google.com/search?q=%235-project-structure-explained)
  - [6. Local Setup & Usage Guide](https://www.google.com/search?q=%236-local-setup--usage-guide)
  - [7. Author & License](https://www.google.com/search?q=%237-author--license)

-----

## 1\. Project Vision & Problem Statement

In today's data-driven world, valuable information is often locked away in unstructured documents like research papers, legal contracts, and technical manuals. Manually searching through these lengthy PDFs is time-consuming and inefficient.

The vision for **DocuReadAI** is to unlock this trapped knowledge. This project demonstrates how a **Retrieval-Augmented Generation (RAG)** architecture can be used to build a powerful assistant that allows users to converse directly with their documents. Instead of reading pages, users can ask specific questions and receive instant, contextually accurate answers, transforming how we interact with and extract value from documents.

-----

## 2\. The RAG Pipeline Explained: How It Works

This application implements a complete, end-to-end RAG workflow. The process ensures that the language model's answers are grounded in the content of the provided document, preventing hallucinations and improving factual accuracy.

### Step 1: Document Loading

The process begins when a user uploads a PDF file through the Gradio interface. **LangChain's `PyPDFLoader`** is used to load the document's content into memory.

### Step 2: Text Splitting

The entire document text is too large to fit into a language model's context window. The **`RecursiveCharacterTextSplitter`** from LangChain breaks the text into smaller, overlapping chunks, preserving semantic context.

### Step 3: Embedding Generation

Each text chunk is converted into a numerical vector, or "embedding," using a **HuggingFace sentence-transformer model** (`all-MiniLM-L6-v2`). These embeddings capture the semantic meaning of the text.

### Step 4: Vector Storage

The generated embeddings and their corresponding text chunks are stored in a **Chroma** vector database. This database runs locally, ensuring data privacy.

### Step 5: Retrieval

When a user asks a question, the query is also converted into an embedding. The system then performs a similarity search in the Chroma DB to find the text chunks that are most semantically relevant to the user's query.

### Step 6: Generation (Answering)

The retrieved text chunks (the "context") are combined with the original question into a prompt. This prompt is then sent to an **IBM Watsonx LLM** (e.g., `google/flan-ul2`), asking it to formulate an answer based *only* on the provided context. This final, grounded answer is then displayed to the user.

-----

## 3\. Core Features & Functionality

  - **Seamless PDF Ingestion:** Users can upload any PDF document, and the application handles all backend processing automatically.
  - **Retrieval-Augmented Generation (RAG):** The core of the application. It ensures answers are factually grounded in the provided document, minimizing hallucinations.
  - **Local & Secure Vector Store:** Uses **Chroma DB** to store document embeddings locally, ensuring user data privacy and fast retrieval.
  - **Enterprise-Grade LLMs:** Leverages the power of **IBM Watsonx** foundation models for reliable and high-quality language generation.
  - **Interactive UI:** A clean and intuitive user interface built with **Gradio** makes the tool accessible to non-technical users.

-----

## 4\. Technical Stack

  - **Orchestration Framework:** **LangChain** – For chaining together all the components of the RAG pipeline.
  - **Embedding Model:** **HuggingFace Transformers** – Utilizes the `all-MiniLM-L6-v2` model for efficient and effective text embedding.
  - **Vector Database:** **Chroma** – A lightweight and fast in-memory vector store for local similarity searches.
  - **Language Model:** **IBM Watsonx** – Provides access to powerful foundation models (e.g., `google/flan-ul2`) for final answer generation.
  - **Web Interface:** **Gradio** – For creating a simple and interactive web-based UI.
  - **Core Language:** **Python**

-----
## Screenshots
<img width="1911" height="862" alt="qa_bot" src="https://github.com/user-attachments/assets/7cd4a4e8-c079-404a-8d16-2b88882d1d1f" />
<img width="547" height="227" alt="load_documents" src="https://github.com/user-attachments/assets/41d778a5-6020-4f67-a995-79cebb0e8b5c" />
<img width="828" height="220" alt="split_text" src="https://github.com/user-attachments/assets/64211367-710d-4731-94fa-225967ce021a" />
<img width="927" height="212" alt="embed_documents" src="https://github.com/user-attachments/assets/974a4a8c-1528-4413-9228-ab0c28a49d12" />
<img width="582" height="191" alt="vector_db" src="https://github.com/user-attachments/assets/6b87bc49-5ff8-4dca-a1ab-b72b892b1859" />
<img width="635" height="210" alt="retriver" src="https://github.com/user-attachments/assets/d07a6e42-5fd3-46eb-8a6e-c37a5153d5c4" />
<img width="4560" height="1113" alt="Blank diagram" src="https://github.com/user-attachments/assets/7b73b50f-7172-4a71-8c75-e6efbdb2c617" />

## 5\. Project Structure Explained

```
.
├── main.py             # A single script containing the full RAG pipeline and Gradio UI logic.
├── .env                # Stores private credentials for the IBM Watsonx API (not tracked by Git).
├── requirements.txt    # Lists all required Python packages for easy installation.
├── .gitignore          # Specifies which files and directories Git should ignore.
└── README.md           # This detailed project documentation.
```

-----

## 6\. Local Setup & Usage Guide

Follow these steps to set up and run the application on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/MrCoss/DocuReadAI.git
cd DocuReadAI
```

### Step 2: Set Up Environment Variables

Create a file named `.env` in the root directory of the project. Open it and add your IBM Watsonx credentials in the following format:

```ini
WATSONX_API_KEY=your_api_key_here
WATSONX_PROJECT_ID=your_project_id_here
WATSONX_URL=https://eu-gb.ml.cloud.ibm.com
MODEL_ID=google/flan-ul2
```

**Important:** Ensure the `.env` file is listed in your `.gitignore` to prevent committing secrets.

### Step 3: Create and Activate a Virtual Environment (Recommended)

```bash
# Create the environment
python -m venv venv

# Activate on Windows
.\venv\Scripts\activate

# Activate on macOS/Linux
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Run the Application

```bash
python main.py
```

After running the script, Gradio will provide a local URL in your terminal (e.g., `http://127.0.0.1:7860`). Open this link in your web browser to start using DocuReadAI.

-----

## 7\. Author & License

  - **Author:** Costas Pinto
  - **GitHub:** [MrCoss](https://github.com/MrCoss)
  - **License:** This project is open-source and available for educational and personal use.
