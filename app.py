__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
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
import streamlit as st
import time
import os
from dotenv import load_dotenv
import base64

load_dotenv()

# Initialize the API key for Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Initialize the Google Generative AI model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
# Load the embeddings model
embeddings = HuggingFaceEmbeddings()

# Define the collection name and persistent directory for Chroma
collection_name = "MLbookcollection"
persist_directory = "MLbookcollection"

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

# Set up the Streamlit page
st.set_page_config(page_title="ML Book Bot")

# Read and encode the sidebar video
with open('src/sidebar_video.mp4', 'rb') as f:
    video_data = f.read()
    encoded_video = base64.b64encode(video_data).decode()

# Apply custom CSS
custom_css = f'''
<style>
/* Background color for main content */
.stApp {{
    background-color: skyblue;
    padding-top: 0px;
}}

/* Ensure chat input box is visible */
.css-1n76uvr.e1tzin5v2 {{
    background-color: rgba(255, 255, 255, 0.8);
    border-radius: 10px;
}}

/* Sidebar styling */
[data-testid="stSidebar"] {{
    position: relative;
    overflow: hidden;
    background: none;
    width: 100%;
    height: 100%;
    padding-top: 0px;
}}

[data-testid="stSidebar"] > div:first-child {{
    position: relative;
    z-index: 1;
}}

[data-testid="stSidebar"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.5); /* Optional: Add overlay */
    z-index: 2;
}}

.sidebar-video {{
    position: absolute;
    top: 0;
    left: 0;
    min-width: 100%;
    min-height: 100%;
    width: auto;
    height: auto;
    z-index: 0;
    object-fit: cover;
}}

/* Center the sidebar title */
.sidebar-content h1 {{
    text-align: center;
}}

</style>
'''

# Embed the CSS and video in the sidebar
st.markdown(custom_css, unsafe_allow_html=True)

# Sidebar with background video
with st.sidebar:
    # Display the video
    sidebar_video_html = f'''
    <div class="sidebar-content">
        <video class="sidebar-video" autoplay loop muted playsinline>
            <source src="data:video/mp4;base64,{encoded_video}" type="video/mp4">
        </video>
        <!-- Sidebar content -->
        <h1>ML Book Bot</h1>
    </div>
    '''
    st.markdown(sidebar_video_html, unsafe_allow_html=True)

# Function for generating LLM response incrementally
def generate_response_stream(user_input):
    response = conversational_rag_chain.invoke(
        {"input": user_input},
        config={
            "configurable": {"session_id": "abc123"}
        },
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

    # Generate a new response
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        for char in generate_response_stream(user_input):
            response_text += char
            response_container.write(response_text)

    message = {"role": "assistant", "content": response_text}
    st.session_state.messages.append(message)
