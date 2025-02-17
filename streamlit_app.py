import streamlit as st
import requests

# Define the Ollama API endpoint
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"

# Streamlit app title
st.title("💬 Health Chatbot (Powered by Ollama API)")
st.write("This chatbot uses the Llama 3 model via the Ollama API.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Create chat input field
if prompt := st.chat_input("What would you like to talk about?"):
    # Store user input
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Send request to Ollama API
    payload = {
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_API_URL, json=payload)
        response.raise_for_status()  # Raise an error if the request fails
        data = response.json()

        if "response" in data:
            reply = data["response"]
        else:
            reply = "⚠️ Error: Unexpected response format from Ollama API."

    except requests.exceptions.RequestException as e:
        reply = f"⚠️ Error: Failed to connect to Ollama API: {str(e)}"
    except ValueError:
        reply = "⚠️ Error: Invalid JSON response from Ollama API."

    # Display Llama 3 response
    with st.chat_message("assistant"):
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
