from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import LLMChain
from typing import List

llmchain = LLMChain()
app=FastAPI()

class link(BaseModel):
    link : List[str]
    
class user_input(BaseModel):
    input:str

@app.post('/query')
async def data_collect(userquery: link):
    data = llmchain.data_collection(userquery.link)
    data = llmchain.data_split(data=data)
    llmchain.vectordb(data=data,add=True)
    return {"output": data}
    

@app.post('/user_query')
async def user_query(input:user_input):
    vectordb = llmchain.vectordb(data=None, add=False)
    ai_output = llmchain.rag(query=input.input,db=vectordb,prompt=llmchain.prompts())
    return {'output':ai_output}