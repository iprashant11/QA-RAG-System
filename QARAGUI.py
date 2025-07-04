#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import re
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# === CONFIGURATION ===
index_folder = r"F:\Prashant Project\Python AI ML Projects\ClassifiedDocsVectorIndex"

# === LOAD EMBEDDINGS & VECTORSTORE ===
embeddings = HuggingFaceEmbeddings(
    model_name='sentence-transformers/all-MiniLM-L6-v2',
    model_kwargs={"device": "cpu"}
)

vectorstore = FAISS.load_local(index_folder, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# === LLM & QA CHAIN ===
llm = Ollama(model="llama3")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === STREAMLIT UI ===
st.set_page_config(page_title="PDF QA", layout="centered")
st.title("🕵️‍♂️ Intelligence Q&A")

question = st.text_input("Ask a question:", "")

if st.button("Get Answer") and question.strip():
    with st.spinner("Thinking..."):
        result = qa_chain(question)

        st.markdown("### 📄 Answer")
        st.write(result['result'])

        relevant_docs_with_scores = vectorstore.similarity_search_with_score(question, k=5)

        st.markdown("### 📚 Source Snippets")
        for i, (doc, score) in enumerate(relevant_docs_with_scores):
            filename = os.path.basename(doc.metadata.get("source", "Unknown"))
            snippet = doc.page_content[:1000]

            # Highlight matched answer
            answer_text = result['result']
            escaped_answer = re.escape(answer_text[:100])
            snippet = re.sub(f"({escaped_answer})", r'<mark style="background-color: yellow">\1</mark>', snippet, flags=re.IGNORECASE)

            with st.expander(f"{i+1}. {filename} (Similarity: {score:.2f})"):
                st.markdown(snippet + "...", unsafe_allow_html=True)

