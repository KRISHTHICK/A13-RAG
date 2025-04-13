from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub  # or use a local LLM
from langchain.chains import RetrievalQA

import os

# Load documents
loader = TextLoader("documents/notes.txt")
documents = loader.load()

# Split the docs into chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=20)
docs = text_splitter.split_documents(documents)

# Use sentence-transformers to embed
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Store in local vector DB (Chroma)
db = Chroma.from_documents(docs, embedding_model)

# Use a simple LLM (offline model) - or use langchain's mock for now
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_id = "microsoft/DialoGPT-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=100)
llm = HuggingFacePipeline(pipeline=pipe)

# Setup RAG
qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

# Chat loop
print("🤖 RAG Chatbot is ready! Type 'exit' to stop.")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        break
    result = qa.run(query)
    print("Bot:", result)
