from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

#how does it know that it needs to create new index if you upload new files
#get files from s3 ???

def get_embeddings():
    embeddings=None
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("Embeddings model is available")
    except Exception as err:
        logging.error(f"Error in creating embeddings: {err}")
    return embeddings

def create_documents():
    texts = None
    try:
        loader = DirectoryLoader("./pdfs/", glob="./*.pdf", loader_cls=PyPDFLoader)
        research_documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(research_documents)
        logging.info("Document are created")
    except Exception as err:
        logging.error(f"Error in document creation: {err}")
    return texts

def create_vector_store():
    embeddings = get_embeddings()
    vector_db = None
    try:
        vector_db = FAISS.load_local("faiss_index", embeddings)
        logging.info("reusing faiss index")
    except:
        try:
            logging.info("creating new vectorstore")
            texts = create_documents()
            vector_db = FAISS.from_documents(
                documents=texts,
                embedding=embeddings
            )
            vector_db.save_local("faiss_index")
            #print(len(texts))
            logging.info("vectorstore was persisted to disc")
        except Exception as err:
            logging.error(err)        
    return vector_db

#TODO: Need to define a function for
    #1. getting and storing model from s3
    #2. getting and storing vector store to s3
    
