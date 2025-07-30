import os
import streamlit as st
import traceback
import shutil

# Langchain imports for conversational memory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Google AI specific imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

# --- Page and App Configuration ---
st.set_page_config(layout="wide")
st.title("Samarth News Research Tool 📈 (True Chat Mode)")

# --- Google AI Configuration and Initialization (runs once) ---
@st.cache_resource
def configure_google_ai():
    """Initializes Google AI models and returns them."""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("🔴 GOOGLE_API_KEY not found in .env file. Please set it to use Google AI.")
        st.stop()
    
    try:
        genai.configure(api_key=google_api_key)
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest", # A powerful and recent model
            temperature=0.7,
            convert_system_message_to_human=True
        )
        embeddings_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        return llm, embeddings_instance
    except Exception as e:
        st.error(f"🔴 Error initializing Google AI components: {e}.")
        print(traceback.format_exc())
        st.stop()

llm, embeddings_instance = configure_google_ai()

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []
# Initialize the memory object for the conversation chain
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True
    )

folder_path = "faiss_index_google"

# --- Sidebar for URL inputs and Controls ---
with st.sidebar:
    st.header("📰 News Article URLs")
    sidebar_urls_input = []
    for i in range(3):
        url = st.text_input(f"URL {i+1}", key=f"url_{i}", help="Enter a news article URL")
        sidebar_urls_input.append(url)

    if st.button("🚀 Process URLs", key="process_urls_button", use_container_width=True):
        with st.spinner("Processing URLs... This may take a moment."):
            urls_to_process = [url for url in sidebar_urls_input if url and url.strip()]
            if not urls_to_process:
                st.warning("No valid URLs provided. Please enter at least one URL.")
            else:
                try:
                    st.info("Loading data from URLs...")
                    loader = UnstructuredURLLoader(
                        urls=urls_to_process,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
                    )
                    data = loader.load()
                    if not data:
                        st.error("No data could be loaded. Check URLs and website accessibility.")
                        st.stop()

                    st.info("Splitting text into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    docs = text_splitter.split_documents(data)
                    if not docs:
                        st.error("Failed to split the documents into processable chunks.")
                        st.stop()

                    st.info("Creating embeddings and FAISS index...")
                    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings_instance)
                    
                    st.info(f"Saving FAISS index to '{folder_path}'...")
                    st.session_state.vectorstore.save_local(folder_path)
                    
                    st.session_state.processed_urls = urls_to_process
                    st.session_state.messages = [{"role": "assistant", "content": f"Successfully processed {len(urls_to_process)} URLs. Ask me anything about them!"}]
                    st.session_state.memory.clear() # Clear memory for the new conversation
                    
                    st.success("✅ URLs processed and ready!")
                    st.rerun()

                except Exception as e:
                    st.error(f"An error occurred during URL processing: {e}")
                    print(traceback.format_exc())

    st.divider()
    if st.button("🔄 Load Existing Index", key="load_index_button", use_container_width=True):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                with st.spinner("Loading existing knowledge base..."):
                    st.session_state.vectorstore = FAISS.load_local(folder_path, embeddings_instance, allow_dangerous_deserialization=True)
                
                st.session_state.messages = [{"role": "assistant", "content": "Loaded existing knowledge base. Ask me anything!"}]
                st.session_state.processed_urls = ["Previously processed"]
                st.session_state.memory.clear() # Clear memory for the new conversation
                
                st.success("✅ Existing index loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load existing index: {e}. It may be corrupted.")
                print(traceback.format_exc())
        else:
            st.warning(f"No existing index found at '{folder_path}'. Please process URLs first.")

    if st.session_state.processed_urls:
        st.subheader("Currently Loaded Articles:")
        for url_item in st.session_state.processed_urls:
            st.caption(f"- {url_item[:70]}..." if len(url_item) > 70 else f"- {url_item}")


# --- Main Chat Interface ---

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a follow-up question:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            st.markdown("I don't have any articles loaded yet. Please process some URLs using the sidebar.")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                try:
                    # Create the conversational chain with memory
                    conversation_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.vectorstore.as_retriever(),
                        memory=st.session_state.memory,
                        return_source_documents=True
                    )

                    # Invoke the chain, which automatically uses the memory
                    result = conversation_chain.invoke({"question": prompt})
                    answer = result["answer"]
                    
                    response_content = answer
                    
                    # Handle the source documents
                    if result.get("source_documents"):
                        response_content += "\n\n**📚 Sources:**\n"
                        sources = set() # Use a set to auto-handle duplicate URLs
                        for doc in result["source_documents"]:
                            if 'source' in doc.metadata:
                                sources.add(doc.metadata['source'])
                        for src in sources:
                            response_content += f"- {src}\n"

                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                except Exception as e:
                    error_message = f"An error occurred: {e}"
                    st.error(error_message)
                    print(traceback.format_exc())
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {error_message}"})
