import base64
import streamlit as st
import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

model = ChatOpenAI(model="gpt-4.1")

st.title("Multi-Image Chat Bot")

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize uploaded images in session state if it doesn't exist
if "uploaded_images" not in st.session_state:
    st.session_state.uploaded_images = []

# File uploader for multiple images
uploaded_files = st.file_uploader("이미지를 업로드해주세요!", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)

# Process and store uploaded images
if uploaded_files:
    st.session_state.uploaded_images = []
    for uploaded_file in uploaded_files:
        st.image(uploaded_file)
        image_bytes = uploaded_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        st.session_state.uploaded_images.append(image_base64)
        uploaded_file.seek(0)  # Reset file pointer for future reads

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("질문을 입력하세요"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Prepare message content with images and text
    message_content = [
        {"type": "text", "text": prompt}
    ]
    
    # Add all uploaded images to the message
    for image_base64 in st.session_state.uploaded_images:
        message_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
        })
    
    # Create and send message to GPT
    message = HumanMessage(content=message_content)
    result = model.invoke([message])
    response = result.content
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)