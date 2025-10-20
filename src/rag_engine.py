# -*- coding: utf-8 -*-
"""
RAG Engine for Technical Assistant.
- Uses a ready-made Q&A dataset (no external dataset required).
- Chroma is used for vector storage.
- Google Gemini 2.0 Flash model is used for answer generation.
- Google's text-embedding-004 model is used for embeddings.
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from data import TECH_QA  # Dataset is stored in a separate file 

# Required libraries: Google GenAI and LangChain components
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# =============== CONFIGURATION ===============
# Get Google API key from .env or system environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Please add GOOGLE_API_KEY to .env or system environment.")

# Configure Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# =============== RAG ENGINE CLASS ===============
class TechnicalAssistantRAG:
    """RAG-based assistant engine that provides Turkish and accurate answers to technical questions."""

    def __init__(self, k: int = 3):
        """
        Initialize RAG engine.
        
        Args:
            k (int): Number of top documents to retrieve for each query (default 3)
        """
        self.vectorstore: Optional[Chroma] = None
        self.retriever: Optional[Chroma] = None
        self.chain: Optional[any] = None
        self.k = k
        self._initialize_rag()  # Start the engine

    def _prepare_documents(self) -> List[Document]:
        """
        Convert the TECH_QA list into Document format understood by LangChain.
        Each item is a text block: "Question: ...\nAnswer: ...".
        """
        if not TECH_QA:
            raise ValueError("TECH_QA dataset is empty. Please provide Q&A entries.")

        docs = []
        for item in TECH_QA:
            content = f"Question: {item['question']}\nAnswer: {item['answer']}"
            docs.append(Document(page_content=content))
        return docs

    def _initialize_rag(self):
        """
        Create Chroma vectorstore and setup the RAG chain.
        Steps:
        1. Define embedding model
        2. Load documents into vectorstore
        3. Setup retriever with k results
        4. Create chain with LLM and prompt template
        """
        # Google's latest embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=GOOGLE_API_KEY
        )

        # Convert dataset to Document format
        docs = self._prepare_documents()

        # Create in-memory Chroma vectorstore 
        self.vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model
        )
        # Retrieve top-k results
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        # Gemini 2.0 Flash model (fast and efficient)
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.3  # Low randomness for technical accuracy
        )

        # Customized prompt for Turkish and safe answers (text remains in Turkish)
        template = """
        Sen, teknik konularda uzmanlaşmış, yardımsever bir yapay zeka asistanısın.
        Kullanıcıya **her zaman Türkçe** yanıt vermelisin.
        Verilen bağlamı (context) dikkatlice incele. Bağlamdaki bilgileri **kendi kelimelerinle, doğal ve akıcı bir şekilde** açıkla.
        Eğer bağlamda sorunun cevabı **tam olarak yoksa ama ilgili bilgi varsa**, o bilgiyi kullanarak **mantıklı ve faydalı bir yanıt oluştur**.
        Sadece bağlamda **hiçbir ilgili bilgi yoksa**, "Bilmiyorum." de.
        Yanıtın **açıklayıcı, dostane ve teknik olarak doğru** olmalı. Gerekiyorsa örnek verebilirsin.
        Asla uydurma veya bağlam dışı iddialarda bulunma.

        Bağlam:
        {context}

        Soru: {question}
        Yanıt:
        """
        prompt = ChatPromptTemplate.from_template(template)

        # Create RAG chain: retriever → prompt → LLM → string output
        self.chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

    def ask(self, query: str) -> str:
        """
        Provide a Turkish answer to the user's technical question using RAG.
        
        Args:
            query (str): User's question (in Turkish)
            
        Returns:
            str: Contextually correct Turkish answer
        """
        if not self.chain:
            raise RuntimeError("RAG engine could not be initialized.")

        try:
            return self.chain.invoke(query)
        except Exception as e:
            # Return a friendly error message in Turkish
            return f"Bilmiyorum. Hata oluştu: {str(e)}"
