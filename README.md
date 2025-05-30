# 🧠 NewsSage AI: Interactive News Research Assistant

**NewsSage AI** is an intelligent assistant designed to extract insights from financial news articles. Leveraging Google's **Gemini 2.0 Flash** model, it allows users to interactively analyze articles, generate embeddings with **OpenAI**, and perform rapid similarity searches using **FAISS**.

![NewsSage AI Interface](https://raw.githubusercontent.com/SamaT-rgb/NewsSage-AI/main/Langchain_prj3.png)

---

## 🚀 Features

- 🔗 Load news article URLs (via direct input or `.txt` file upload)
- 📄 Extract and process article content using **LangChain** + **Unstructured**
- 🧠 Generate vector embeddings using **OpenAI**
- ⚡ Store and retrieve relevant article chunks using **FAISS**
- 💬 Chat with the AI about the loaded news articles
- 🧾 Ask contextual questions and get intelligent, referenced answers
- 💾 Save and reuse FAISS index for fast future querying

---

## 🛠️ Installation

### 1. Clone the Repository
git clone https://github.com/SamaT-rgb/NewsSage-AI.git
cd NewsSage-AI

### 2. Set Up Python Virtual Environment
Ensure you have Python 3.8 or higher installed.
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
### 3. Install Python Dependencies
pip install -r requirements.txt

### 4. Set Up Environment Variables
Create a .env file in the root directory and add your Google API key:
GOOGLE_API_KEY=your_google_api_key_here

### 5. Run the Application
streamlit run app.py
After a few seconds, the application should be accessible at:
📍 http://localhost:8501/

### 📂 Project Structure
NewsSage-AI/
├── app.py
├── requirements.txt
├── .env
├── Langchain_prj1.png
├── faiss_index_google/
└── ...
### 📝 Notes
Replace your_google_api_key_here with your actual Google API key.

Ensure that all dependencies are installed before running the application.

The FAISS index is saved in the faiss_index_google/ directory for future use.
