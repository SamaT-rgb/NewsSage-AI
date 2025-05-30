import os
import streamlit as st
import time # Keep for any UI delays you might want
import traceback
import shutil # For removing directory if FAISS load fails

# Langchain imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader # Corrected import
from langchain_community.vectorstores import FAISS

# Google AI specific imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

st.set_page_config(layout="wide") # Use a wider layout for chat
st.title("Samarth News Research Tool 📈 (Chat Mode)")

# --- Google AI Configuration and Initialization (runs once) ---
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    st.error("🔴 GOOGLE_API_KEY not found in .env file. Please set it to use Google AI.")
    st.stop()
else:
    try:
        genai.configure(api_key=google_api_key)
    except Exception as e:
        st.error(f"🔴 Error configuring Google AI SDK: {e}. Ensure the key is valid.")
        print(traceback.format_exc())
        st.stop()

# Initialize LLM and Embeddings - these can be global as they don't change per session
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", # Using a more recent model if available
        temperature=0.7,
        # max_output_tokens=2048, # Gemini models often have larger context/output
        convert_system_message_to_human=True
    )
    embeddings_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Error initializing Google AI components: {e}. "
             "Check API key, permissions, and installed packages.")
    print(traceback.format_exc())
    st.stop()

# --- Session State Initialization ---
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = [] # To store chat history {role: "user/assistant", content: "..."}
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = [] # To show which URLs are currently loaded

folder_path = "faiss_index_google" # Folder to save/load FAISS index

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
                        st.error("No data could be loaded from the provided URLs. Check URLs and website accessibility.")
                        st.stop()

                    st.info("Splitting text into chunks...")
                    text_splitter = RecursiveCharacterTextSplitter(
                        separators=['\n\n', '\n', '.', ','],
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    docs = text_splitter.split_documents(data)
                    if not docs:
                        st.error("Failed to split the documents into processable chunks.")
                        st.stop()

                    st.info("Creating embeddings and FAISS index...")
                    st.session_state.vectorstore = FAISS.from_documents(docs, embeddings_instance)
                    
                    st.info(f"Saving FAISS index to '{folder_path}'...")
                    st.session_state.vectorstore.save_local(folder_path)
                    
                    st.session_state.processed_urls = urls_to_process
                    st.session_state.messages = [{"role": "assistant", "content": f"Successfully processed {len(urls_to_process)} URLs. Ask me anything about them!"}] # Reset chat
                    st.success("✅ URLs processed and ready!")
                    st.rerun() # Rerun to update main chat interface

                except Exception as e:
                    st.error(f"An error occurred during URL processing: {e}")
                    print(traceback.format_exc())

    st.divider()
    if st.button("🔄 Load Existing Index", key="load_index_button", use_container_width=True):
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            try:
                with st.spinner("Loading existing knowledge base..."):
                    st.session_state.vectorstore = FAISS.load_local(
                        folder_path,
                        embeddings_instance,
                        allow_dangerous_deserialization=True # Be cautious with this in untrusted environments
                    )
                st.session_state.messages = [{"role": "assistant", "content": "Loaded existing knowledge base. Ask me anything!"}]
                # Try to infer processed URLs if possible (e.g., store them with the index, or just show a generic message)
                st.session_state.processed_urls = ["Previously processed (details not stored with index)"]
                st.success("✅ Existing index loaded!")
                st.rerun()
            except Exception as e:
                st.error(f"Could not load existing index: {e}. The index might be corrupted or incompatible.")
                print(traceback.format_exc())
                # Optionally remove corrupted index
                # if st.checkbox("Delete corrupted index and retry processing?"):
                #     try:
                #         shutil.rmtree(folder_path)
                #         st.warning(f"Removed index at '{folder_path}'. Please process URLs again.")
                #     except Exception as rm_e:
                #         st.error(f"Failed to remove index folder: {rm_e}")
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
if prompt := st.chat_input("Ask a question about the articles:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if vectorstore is loaded
    if st.session_state.vectorstore is None:
        with st.chat_message("assistant"):
            response = "I don't have any articles loaded yet. Please process some URLs using the sidebar."
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... 🤔"):
                try:
                    chain = RetrievalQAWithSourcesChain.from_llm(
                        llm=llm,
                        retriever=st.session_state.vectorstore.as_retriever()
                    )
                    result = chain.invoke({"question": prompt}) # Use invoke for newer Langchain
                    # result will be a dictionary, e.g. --> {"question": ..., "answer": ..., "sources": ... }
                    
                    answer = result.get("answer", "Sorry, I couldn't find an answer.")
                    sources = result.get("sources", "")

                    response_content = f"{answer}"
                    if sources:
                        # Format sources nicely
                        sources_list = []
                        if isinstance(sources, str):
                            sources_list = [s.strip() for s in sources.split('\n') if s.strip()]
                        elif isinstance(sources, list): # Some chains might return a list
                            sources_list = [str(s).strip() for s in sources if str(s).strip()]
                        
                        if sources_list:
                            response_content += "\n\n**📚 Sources:**\n"
                            for src_item in list(set(sources_list)): # Show unique sources
                                response_content += f"- {src_item}\n"
                    
                    st.markdown(response_content)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})

                except Exception as e:
                    error_message = f"An error occurred while searching for an answer: {e}"
                    st.error(error_message)
                    print(traceback.format_exc())
                    st.session_state.messages.append({"role": "assistant", "content": f"⚠️ Error: {error_message}"})