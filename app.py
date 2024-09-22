import sys
import os
import time
import base64
from dotenv import load_dotenv
import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables
load_dotenv()

# Replace sqlite3 with pysqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Initialize the API key for Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
# Load the embeddings model
embeddings = HuggingFaceEmbeddings()

# Define the collection name and persistent directory for Chroma
collection_name = "MLbookcollection"
persist_directory = "MLbookcollection"  # Specify the persistent directory

# Initialize Chroma by loading the existing collection
knowledge = Chroma(
    collection_name=collection_name,
    persist_directory=persist_directory,
    embedding_function=embeddings
)
retriever = knowledge.as_retriever()

### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    '''
    This is the data from the book named "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow 3rd Edition". I give you access to all the data in this book. Whatever question is asked, you have to answer it properly, comprehensively, and in detail. Whenever a question is asked from this book, you always have to answer the question in English language no matter if in prompt it mentions to answer in English or not, but if it specifies to answer in some other language, only then you have to change the language in giving a response.
    One thing also u have to remember that your owner and the one who created u is "Mr Hamza Aleem" which is student of electrical engineering at "NUST" in Pakistan,and he is expert in microwave angineering and antennae design.
    '''
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

### Statefully manage chat history ###
if 'store' not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to inject custom CSS
def inject_custom_css():
    # Add background image to sidebar
    with open("src/backgroundpic.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    sidebar_style = f"""
    <style>
    /* Sidebar background image */
    .css-1d391kg {{
        background-image: url("data:image/jpeg;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    /* Adjust sidebar text color for better visibility */
    .css-1d391kg .sidebar-content {{
        color: white;
    }}
    /* Main chat area background color */
    .block-container {{
        background-color: #87CEEB; /* Sky Blue */
    }}
    </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)

# Inject the custom CSS
inject_custom_css()

# Create an instance of the ChatBot class
st.set_page_config(page_title="HandsOn Machine Learning with ScikitLearn Keras and TensorFlow 3rd Edition")

with st.sidebar:
    st.title('HandsOn Machine Learning with ScikitLearn Keras and TensorFlow 3rd Edition')

# Function for generating LLM response incrementally
def generate_response_stream(user_input):
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": "abc123"}
        },  # constructs a key "abc123" in `store`.
    )["answer"]
    # Simulate streaming by yielding one character at a time
    for char in response:
        yield char
        time.sleep(0.005)  # Adjust this to control the typing speed

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask me anything from your book"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Generate a new response if the last message is not from the assistant
    with st.chat_message("assistant"):
        response_container = st.empty()  # Create an empty container for streaming the response
        response_text = ""

        for char in generate_response_stream(user_input):
            response_text += char
            response_container.write(response_text)

    message = {"role": "assistant", "content": response_text}
    st.session_state.messages.append(message)
