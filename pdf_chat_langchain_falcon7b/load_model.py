import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain import HuggingFacePipeline
import logging

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file



def download_model():
    tokenizer, base_model = None, None
    try:
        repo_id = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(repo_id)
        base_model = AutoModelForCausalLM.from_pretrained(repo_id, load_in_8bit=True, device_map="auto", trust_remote_code=True)
        logging.info("Model is available")
    except Exception as err:
        logging.error(f"Couldn't download model: {err}")
    return tokenizer, base_model

def create_hf_pipeline(tokenizer, base_model):
    llm = None
    try:
        pipe = transformers.pipeline(
            "text-generation",
            model=base_model,
            tokenizer=tokenizer,
            max_length=512,
            max_new_tokens=1500,
            do_sample=True,
            num_return_sequences=1,
            eos_token_id = tokenizer.eos_token_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        logging.info("transformers pipeline created")
        llm = HuggingFacePipeline(pipeline=pipe, model_kwargs={"temprature": 0.1})
        logging.info("HuggingFace pipeline created")
    except Exception as err:
        logging.info(f"Couldn't create llm: {err}")
    return llm
