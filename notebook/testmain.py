import os
import google.generativeai as gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader,WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers.string import StrOutputParser
from dotenv import load_dotenv
import tqdm as notebook_tqdm

load_dotenv()

# --------------------------------API CONFIGURATION-----------------------
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
key = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=key)
google_embed = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=key)

# ----------------------------LLMChain---------------------------------------
class LLMChain:
    def __init__(self):
        pass
        
    # Data collection
    def data_collection(self, data):
        url_data = WebBaseLoader(data,continue_on_failure=True,verify_ssl=False,requests_per_second=1)
        return url_data.load()

    # Data splitting
    def data_split(self, data):
        split = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=250, length_function=len)
        splited_data = split.split_documents(data)
        return splited_data

    # Vector database creation
    def vectordb(self,data,add=False):
        if not os.path.exists("./chromadb_vector"):
            vector_store = Chroma.from_documents(documents=data, embedding=google_embed, persist_directory="./chromadb_vector")
        else:
            vector_store = Chroma(persist_directory="./chromadb_vector", embedding_function=google_embed)
            if add==True:
                vector_store.add_documents(documents=data)
        return vector_store
    
    # Test retrieval
    def test_retrival(self,db, query, n=3):
        retrive_test = db.similarity_search(query, k=n)
        return f"Retrival Test Successful --> {retrive_test}"
    
    # Prompt setup
    def prompts(self):
        prompt = PromptTemplate(
            template="You are a super advanced AI which has knowledge of everything. Generate info on the basis of the given context and also add your creativity and explain like I am five. Context: {context}\nQuestion: {question}\nAnswer:",
            input_variables=['context', 'question']
        )
        return prompt

    # Retrieval-Augmented Generation (RAG) chain
    def rag(self, query,db,prompt):
        retriver = db.as_retriever()
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        rag_chain = (
            {'context': retriver | format_docs, 'question': RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )
        
        output = rag_chain.invoke(query)
        return output

if __name__ == '__main__':
    pass
