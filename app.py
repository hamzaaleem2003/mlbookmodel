import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

class ChatBot:
    def __init__(self):
        # Initialize the API key for Google Generative AI
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Load the embeddings model
        embeddings = HuggingFaceEmbeddings()
        
        # Define the collection name and persistent directory for Chroma
        collection_name = "MLbookcollection"
        persist_directory = "MLbookcollection"

        # Initialize Chroma with the embedding function
        self.knowledge = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=embeddings
        )

        # Initialize the Google Generative AI model
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
        
        # Define the template for prompting
        self.template = """
        this is the data from the book and name of the book is "HandsOn Machine Learning with ScikitLearn Keras and TensorFlow 3rd Edition". I give you access to all the data in this book. Whatever question is asked you have to answer that properly and comprehensively and in detail. Whenever a question is asked from this book, you always have to answer the question in English language. However, if it specifies to answer in some other language, only then you change the language in giving a response.

        Context: {context}

        Question: {question}

        Answer:
        """
        
        # Initialize the prompt template
        self.prompt = PromptTemplate(
            template=self.template,
            input_variables=["context", "question"]
        )

        # Define the RAG chain for retrieval and generation
        self.rag_chain = (
            {"context": self.knowledge.as_retriever(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    def generate_response(self, input, context):
        # Extend the existing context with the new question
        updated_context = (context + " " + input).strip()
        # Invoke the RAG chain with updated context
        response = self.rag_chain.invoke({"context": updated_context, "question": input})
        # Return both the response and the new context
        return response, updated_context

# Create an instance of the ChatBot class
# Create an instance of the ChatBot class
bot = ChatBot()
st.set_page_config(page_title="ML Book Bot")
with st.sidebar:
    st.title('ML Book Bot')

# Initialize session state for messages and context if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask me anything from your book"}]
if "context" not in st.session_state:
    st.session_state.context = ""

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
user_input = st.chat_input()
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Ensure context is a string
    current_context = st.session_state.context if isinstance(st.session_state.context, str) else ""
    response, updated_context = bot.generate_response(user_input, current_context)
    st.session_state.context = updated_context  # Update the context in the session state

    st.session_state.messages.append({"role": "assistant", "content": response})
