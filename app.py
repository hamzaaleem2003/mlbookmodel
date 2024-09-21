__import__('pysqlite3')  # Dynamically imports the pysqlite3 module
import sys  # Imports the sys module necessary to modify system properties
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')  # Replaces the sqlite3 entry in sys.modules with pysqlite3
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import streamlit as st
import time
import os
from dotenv import load_dotenv
load_dotenv()



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
                this is the data from the book and name of the book is "HandsOn Machine Learning with ScikitLearn Keras and TensorFlow 3rd Edition", I give u access to all the data in this book , whatever question is asked you have to answer that properly and comprehensively and in detail, whenever a question is asked from this book you always have to answer the question in English language no matter if in prompt it mentions to answer in English or not, but if it specifies to answer in some other language, only then you have to change the language in giving a response.
                '''
                "\n\n"
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
store = {}
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
      store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
# Create an instance of the ChatBot class
st.set_page_config(page_title="ML Book Bot")
with st.sidebar:
    st.title(' ML Book Bot')

# Function for generating LLM response incrementally
def generate_response_stream(input):
    response = conversational_rag_chain.invoke(
        {"input": input},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
    )["answer"]
    # Simulate streaming by yielding one character at a time
    for char in response:
        yield char
        time.sleep(0.005)  # Adjust this to control the typing speed

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Welcome, ask me anything from your book"}]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User-provided prompt
if input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": input})
    with st.chat_message("user"):
        st.write(input)

# Generate a new response if the last message is not from the assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response_container = st.empty()  # Create an empty container for streaming the response
        response_text = ""

        for char in generate_response_stream(input):
            response_text += char
            response_container.write(response_text)

    message = {"role": "assistant", "content": response_text}
    st.session_state.messages.append(message)

