import streamlit as st
import os, time
from datetime import datetime
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama


# Initialize local llama3 model from Ollama
llm = Ollama(model="llama3")

# Load Chroma vector store
CHROMA_DB_PATH = "chroma_db"
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embedding)

# Streamlit app configuration
st.set_page_config(page_title="Cancer Knowledge AI Assistant", page_icon="💬")
st.title("Cancer Knowledge AI Assistant 💡")
st.markdown("**This assistant offers information drawn from World Health Organization (WHO) and National Cancer Institute (NCI) websites.**")

# Initialize session state for countdown and save button
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "show_save_button" not in st.session_state:
    st.session_state.show_save_button = False

# Chat style selection
style = st.selectbox("Select a Chatbot", ("Chatbot A", "Chatbot B"))
if style == "Chatbot A":
    style_prefix = (
        "You are an empathetic assistant. Respond with compassion, care, and emotional understanding. "
    )
else:
    style_prefix = (
        "You are a neutral, factual cancer knowledge assistant. Respond directly and objectively. "
    )

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hello! I am a **Cancer Knowledge AI Assistant**. Ask me anything!"
        }
    ]

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Capture user input
query = st.chat_input("Ask me anything about cancer or health...")

if query:
    # Add user's message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # 1. Retrieve relevant documents using vector similarity
    retrieved_docs = vectorstore.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in retrieved_docs])

    # 2. Build the prompt with retrieved context and selected style
    prompt = f"{style_prefix}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"

    # 3. Generate response from the local llama3 model
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = llm.invoke(prompt)
            st.markdown(answer)
            # Add assistant's reply to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Countdown timer and show save button after 5 minutes
elapsed = time.time() - st.session_state.start_time
if elapsed > 300:
    st.session_state.show_save_button = True

# Save chat history to local file
if st.session_state.show_save_button:
    if st.button("💾 Save Chat History"):
        os.makedirs("chat_logs", exist_ok=True)
        filename = f"chat_logs/chat_{style}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            for msg in st.session_state.messages:
                f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
        st.success(f"✅ Saved to `{filename}`")
        st.warning(
            "**This conversation was part of a scientific study.** "
            "The chatbot’s responses are generated based on existing public health sources but **should not** be interpreted as personalized medical advice. "
            "If you have questions about your health, please consult a healthcare professional."
        )
