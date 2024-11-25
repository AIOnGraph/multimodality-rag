import streamlit as st
from utilis import create_document_and_vectorstore, get_response_from_llm
import os
import logging
import tempfile
import base64
from io import BytesIO
from PIL import Image


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

output_path = "figures/"
UPLOAD_DIR = "uploads/"
os.makedirs(UPLOAD_DIR, exist_ok=True)

try:
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        logger.info(f"Created directory: {output_path}")
except OSError as error:
    logger.error(f"Error creating directory {output_path}: {error}")

st.set_page_config(initial_sidebar_state='collapsed', layout='wide')
st.title("**Multi-Modality RAG**")
subheader = st.subheader("Enter the OpenAI API key in the sidebar to chat with your document.", divider=True)

with st.sidebar:
    st.header("Multi-Modality RAG")
    st.markdown(
        """
        ### How to use
        1. Enter your OpenAI API key below🔑
        2. Upload a PDF📄
        3. Ask a question about the document💬
        """
    )
    OPENAI_API_KEY = st.text_input("OPENAI_API_KEY", type="password")

if OPENAI_API_KEY:
    logger.info("OpenAI API key provided.")
    subheader.empty()
    uploaded_pdf = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_pdf:
        pdf_path = os.path.join(UPLOAD_DIR, uploaded_pdf.name)  # Create a path for the uploaded file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())  # Save the uploaded PDF to the specified path
        logger.info(f"PDF saved to: {pdf_path}")
        logger.info(f"PDF uploaded: {uploaded_pdf.name}")
        logger.info(f"PDF saved to: {temp_pdf_path}")
        try:
            vectorstore = create_document_and_vectorstore(OPENAI_API_KEY, pdf_path, output_path)
            logger.info("Vectorstore created successfully.")

            if vectorstore:
                if 'messages' not in st.session_state:
                    st.session_state['messages'] = []
                    logger.debug("Initialized session state for messages.")

                for message in st.session_state.messages:
                    if message["role"] == 'assistant':
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])
                    else:
                        with st.chat_message(message["role"]):
                            st.markdown(message["content"])

                if query := st.chat_input("Ask me anything"):
                    logger.info(f"User query: {query}")
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)

                    with st.chat_message("assistant"):
                        message_placeholder = st.empty()
                        try:
                            result, relevant_image = get_response_from_llm(vectorstore, query, OPENAI_API_KEY)
                            logger.info("Response generated successfully.")
                            st.write(result)
                            if relevant_image:
                                for image in relevant_image:
                                    image_data = base64.b64decode(image)
                                    image = Image.open(BytesIO(image_data))

                                    st.image(image)
                                    break
                            st.session_state.messages.append({"role": "assistant", "content": result})
                        except Exception as e:
                            logger.error(f"Error generating response: {e}")
                            st.error("There was an error generating the response. Please try again.")
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            st.error("There was an error processing the PDF. Please try again.")
else:
    logger.warning("No OpenAI API key provided.")
