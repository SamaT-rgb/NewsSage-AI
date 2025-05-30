import os
import streamlit as st
import pickle
import time # Keep for potential UX delays, though less critical with status updates
import traceback # To print full tracebacks for debugging

# Langchain imports - ensure you have the latest versions for these paths
from langchain_openai import OpenAI # Updated import for newer Langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Corrected import for UnstructuredURLLoader
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings # Updated import for newer Langchain
from langchain_community.vectorstores import FAISS # Updated import for newer Langchain

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

sidebar_urls = []
for i in range(3):
    # Add a unique key for each text_input widget for better state management
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    sidebar_urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()

# Initialize LLM
# This should be done early. If it fails, the app shouldn't proceed.
try:
    llm = OpenAI(temperature=0.9, max_tokens=500)
except Exception as e:
    st.error(f"Error initializing OpenAI LLM: {e}. "
             "Please ensure your OPENAI_API_KEY is set correctly in the .env file and is valid.")
    print(traceback.format_exc())
    st.stop() # Critical failure, stop the app

if process_url_clicked:
    # Filter out empty or whitespace-only URLs
    urls_to_process = [url for url in sidebar_urls if url and url.strip()]

    if not urls_to_process:
        main_placeholder.error("No valid URLs provided. Please enter at least one URL in the sidebar.")
        st.stop()

    main_placeholder.info("Processing URLs... This may take a moment. 🚀")
    
    # 1. Load data from URLs
    # Adding User-Agent can help with sites that block default Python user-agents.
    loader = UnstructuredURLLoader(
        urls=urls_to_process, 
        # ssl_verify=False, # Uncomment if you face SSL issues with specific sites
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"} # Using a common user-agent
    )
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
    except Exception as e:
        main_placeholder.error(f"Error loading data from URLs: {e}")
        print(f"Detailed error during data loading:\n{traceback.format_exc()}")
        st.stop()
    
    if not data:
        main_placeholder.error("No data could be loaded from the provided URLs. "
                               "Please check if the URLs are correct, accessible, "
                               "and contain readable textual content. Some websites may block automated scraping.")
        st.stop()

    # 2. Split data into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200  # Adding overlap can improve context retention between chunks
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    if not docs:
        # This can happen if documents are empty or too small to be split meaningfully
        main_placeholder.error("Failed to split the documents into processable chunks. "
                               "The content retrieved might be too small, empty, or not in a format "
                               "that can be effectively processed by the text splitter.")
        st.stop()

    # 3. Create embeddings and FAISS index
    main_placeholder.text("Creating Embeddings and FAISS Index...Started...✅✅✅")
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        main_placeholder.error(f"Error creating embeddings or FAISS index: {e}. "
                               "This could be due to an issue with the OpenAI API (e.g., invalid key, "
                               "insufficient quota, rate limits) or a problem with the FAISS library.")
        print(f"Detailed error during embedding/FAISS creation:\n{traceback.format_exc()}")
        st.stop()
    
    main_placeholder.text("Embedding and Indexing complete. Saving to file...✅✅✅")

    # 4. Save the FAISS index to a pickle file
    try:
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore_openai, f)
        main_placeholder.success("✨ Success! URLs processed and FAISS index saved. Ready for your questions! ✨")
        # st.balloons() # Optional: for a more visual success indication
        # Stop script execution here to ensure the success message is displayed.
        # The query input will appear on the next user interaction/rerun.
        st.stop() 
    except Exception as e:
        main_placeholder.error(f"Error saving FAISS index to file: {e}")
        print(f"Detailed error during FAISS saving:\n{traceback.format_exc()}")
        st.stop()

# ----- Query Section -----
# This part runs if process_url_clicked is False, or on a rerun after st.stop().
# The main_placeholder will be replaced by the text_input widget.
query = main_placeholder.text_input("Ask a question about the content of the URLs:")

if query:
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)
            
            # llm should be available as it's initialized globally and app stops if it fails.
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            
            with st.spinner("Searching for answers... 🧐"): # Show a spinner during processing
                result = chain({"question": query}, return_only_outputs=True)
            
            st.header("📝 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources:")
                # Sources can be a string or a list, handle both.
                if isinstance(sources, str):
                    sources_list = sources.split("\n")
                elif isinstance(sources, list):
                    sources_list = sources
                else: # Fallback for unexpected format
                    sources_list = [str(sources)] 
                
                for source_item in sources_list:
                    if source_item.strip(): # Avoid writing empty lines
                        st.write(source_item.strip())

        except Exception as e:
            st.error(f"An error occurred during question answering: {e}")
            print(f"Detailed error during QA processing:\n{traceback.format_exc()}")
    else:
        # If the FAISS file doesn't exist, guide the user.
        st.warning("FAISS index file not found. Please process URLs first using the sidebar.")