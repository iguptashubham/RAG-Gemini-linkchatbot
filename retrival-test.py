from main import LLMChain

llmchain = LLMChain()

vectordb = llmchain.vectordb(data=None,add=False)
llmchain.test_retrival(db=vectordb, query='what is Lightllm')