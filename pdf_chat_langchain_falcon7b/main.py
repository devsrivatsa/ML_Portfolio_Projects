from fastapi import (
    FastAPI, 
    Response, 
    status, 
    Form
)
from load_model import download_model, create_hf_pipeline
from prepare_data import create_vector_store
from langchain.chains.question_answering import load_qa_chain
import logging

def prepare():
    #1. download model and create llm pipeline
    tokenizer, model = download_model()
    llm = create_hf_pipeline(tokenizer, model)
    #2. Create vector store of doccuments
    vector_db = create_vector_store()
    #3. create llm chain
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
    return llm, vector_db, qa_chain

app = FastAPI()

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]_[%(funcName)s]: %(message)s")
global llm, vector_db, qa_chain
llm, vector_db, qa_chain = [None]*3 

# 1. download model and create llm pipeline
tokenizer, model = download_model()
llm = create_hf_pipeline(tokenizer, model)

#2. Create vector store of doccuments
vector_db = create_vector_store()

#3. create llm chain
qa_chain = load_qa_chain(llm=llm, chain_type="stuff")


@app.get("/health_check")
async def health_check(res:Response):
    if llm and vector_db and qa_chain:
        res.status_code = 200
        return { "msg": "Model is loaded and vector store is ready!" }
    else:
        try:
            prepare()
            res.status_code = 201
            return { "msg": "Model and/or vector store is/are creeated!" }
        except Exception as exp:
            res.status_code = 500
            return { "msg": f"something wrong: {exp}" }

@app.post("/prompt")
async def predict(question:str = Form()):
    answer = qa_chain.run(
        input_documents=vector_db.similarity_search(question),
        question=question
    )
    return { "model_response": answer }



    
