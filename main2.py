import os
import streamlit as st
# import pickle # REMOVE - We'll use FAISS's own saving mechanism
import time
import traceback

# Langchain imports
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS

# Google AI specific imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

st.title("RockyBot: News Research Tool 📈 (Powered by Google AI)")
st.sidebar.title("News Article URLs")

sidebar_urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    sidebar_urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
folder_path = "faiss_index_google"

main_placeholder = st.empty() # For status updates during processing AND for the query input

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

llm = None
embeddings_instance = None

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.7,
        max_output_tokens=500,
        convert_system_message_to_human=True
    )
    embeddings_instance = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Error initializing Google AI components: {e}. "
             "Check API key, permissions, and installed packages.")
    print(traceback.format_exc())
    st.stop()

if process_url_clicked:
    urls_to_process = [url for url in sidebar_urls if url and url.strip()]

    if not urls_to_process:
        main_placeholder.error("No valid URLs provided.")
        st.stop() # Stop if no URLs, query input won't show, which is correct

    main_placeholder.info("Processing URLs... 🚀")
    
    loader = UnstructuredURLLoader(
        urls=urls_to_process,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    )
    main_placeholder.text("Data Loading...Started...✅✅✅")
    try:
        data = loader.load()
    except Exception as e:
        main_placeholder.error(f"Error loading data: {e}")
        print(traceback.format_exc())
        st.stop() # Stop on error, query input won't show
    
    if not data:
        main_placeholder.error("No data loaded from URLs.")
        st.stop() # Stop if no data, query input won't show

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=200
    )
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    if not docs:
        main_placeholder.error("Failed to split documents.")
        st.stop() # Stop if splitting fails

    main_placeholder.text("Creating Embeddings (Google AI) and FAISS Index...Started...✅✅✅")
    try:
        vectorstore_google = FAISS.from_documents(docs, embeddings_instance)
    except Exception as e:
        main_placeholder.error(f"Error creating embeddings/FAISS index: {e}.")
        print(traceback.format_exc())
        st.stop() # Stop on error
    
    main_placeholder.text("Embedding and Indexing complete. Saving to local folder...✅✅✅")

    try:
        vectorstore_google.save_local(folder_path)
        # CHANGED: Use st.success for a persistent message, and REMOVE st.stop()
        st.success(f"✨ Success! FAISS index saved to folder '{folder_path}'. Ready for questions! ✨")
        # NO st.stop() here. Script will continue to the query input part.
    except Exception as e:
        main_placeholder.error(f"Error saving FAISS index locally: {e}")
        print(f"Detailed error during FAISS saving:\n{traceback.format_exc()}")
        st.stop() # Stop if saving fails

# ----- Query Section -----
# This section will now ALWAYS run unless st.stop() was called earlier due to an error
# or if process_url_clicked was True and an error occurred within that block.
# If process_url_clicked was True and successful, the st.success message will be shown
# and THEN this query input will appear using main_placeholder.
query = main_placeholder.text_input("Ask a question about the content of the URLs (using Google AI):", key="query_input_main") # Added a unique key

if query:
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        try:
            if embeddings_instance is None:
                st.error("Embeddings model not initialized. Cannot load FAISS index.")
                st.stop()
            
            # It's good practice to display a spinner or message while loading the index too.
            with st.spinner("Loading knowledge base..."):
                vectorstore = FAISS.load_local(folder_path, embeddings_instance, allow_dangerous_deserialization=True)

            if llm is None:
                 st.error("LLM not available.")
                 st.stop()

            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            
            with st.spinner("Searching for answers with Google AI... 🧐"):
                result = chain({"question": query}, return_only_outputs=True)
            
            st.header("📝 Answer")
            st.write(result["answer"])

            sources = result.get("sources", "")
            if sources:
                st.subheader("📚 Sources:")
                if isinstance(sources, str):
                    sources_list = sources.split("\n")
                elif isinstance(sources, list):
                    sources_list = sources
                else:
                    sources_list = [str(sources)] 
                
                for source_item in sources_list:
                    if source_item.strip():
                        st.write(source_item.strip())

        except Exception as e:
            st.error(f"An error occurred during question answering: {e}")
            print(f"Detailed error during QA processing:\n{traceback.format_exc()}")
    else:
        # This message will now show if folder_path doesn't exist,
        # for example, on the first run before any URLs are processed.
        # The main_placeholder would have rendered the text_input already.
        # To avoid main_placeholder showing the input AND this warning,
        # we might want to conditionally show the input OR this warning.
        # For now, let's keep it simple. The input will be there, and this warning
        # will appear below it if the folder is missing.
        st.warning(f"FAISS index folder '{folder_path}' not found. Please process URLs first using the sidebar.")