import os
import uuid
import base64
from unstructured.partition.pdf import partition_pdf
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import logging
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_raw_pdf_elements(pdf_path, output_path):
    logger.info(f"Extracting elements from PDF: {pdf_path}")
    try:
        raw_pdf_pages = partition_pdf(
            filename=pdf_path,
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=3800,
            combine_text_under_n_chars=2000,
            image_output_dir_path=output_path,
        )
        logger.info("Successfully extracted elements from the PDF.")
        return raw_pdf_pages
    except Exception as e:
        logger.error(f"Error while extracting PDF elements: {e}")
        raise

def summarize_text(openai_api_key, raw_pdf_elements):
    logger.info("Starting text and table summarization.")
    text_elements = []
    table_elements = []

    text_summaries = []
    table_summaries = []
    summary_prompt = """
    Summarize the following {element_type}:
    {element}
    """
    try:
        summary_chain = LLMChain(
            llm=ChatOpenAI(
                model="gpt-4o-mini", openai_api_key=openai_api_key, max_tokens=1024
            ),
            prompt=PromptTemplate.from_template(summary_prompt),
        )
        for e in raw_pdf_elements:
            if "CompositeElement" in repr(e):
                logger.debug("Summarizing a text element.")
                text_elements.append(e.text)
                summary = summary_chain.run({"element_type": "text", "element": e})
                text_summaries.append(summary)

            elif "Table" in repr(e):
                logger.debug("Summarizing a table element.")
                table_elements.append(e.text)
                summary = summary_chain.run({"element_type": "table", "element": e})
                table_summaries.append(summary)
        
        logger.info("Text and table summarization completed successfully.")
    except Exception as e:
        logger.error(f"Error during text or table summarization: {e}")
        raise
    return text_elements, text_summaries, table_elements, table_summaries

def encode_image(image_path):
    logger.info(f"Encoding image: {image_path}")
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Error encoding image: {e}")
        raise

def summarize_image(encoded_image, openai_api_key):
    logger.debug("Starting image summarization.")
    prompt = [
        SystemMessage(content="You are a bot that is good at analyzing images."),
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe the contents of this image."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"},
                },
            ]
        ),
    ]
    try:
        response = ChatOpenAI(
            model="gpt-4-vision-preview", openai_api_key=openai_api_key, max_tokens=1024
        ).invoke(prompt)
        logger.info("Image summarization completed.")
        return response.content
    except Exception as e:
        logger.error(f"Error during image summarization: {e}")
        raise

def get_image_summaries(openai_api_key, output_path):
    logger.info("Gathering image summaries.",11111111111111)
    image_elements = []
    image_summaries = []
    print(os.listdir(output_path),55555555555555555555555555555)
    for i in os.listdir(output_path):
        print(i,111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111)
        if i.endswith((".png", ".jpg", ".jpeg")):
            logger.debug(f"Processing image: {i}")
            image_path = os.path.join(output_path, i)
            encoded_image = encode_image(image_path)
            image_elements.append(encoded_image)
            summary = summarize_image(encoded_image, openai_api_key)
            image_summaries.append(summary)
    logger.info("Image summaries collected successfully.")
    return image_elements, image_summaries

def create_document_and_vectorstore(openai_api_key, pdf_path, output_path):
    logger.info(f"Creating document and vectorstore for PDF: {pdf_path}")
    try:
        raw_pdf_elements = get_raw_pdf_elements(pdf_path, output_path)
        text_elements, text_summaries, table_elements, table_summaries = summarize_text(
            openai_api_key, raw_pdf_elements
        )
        image_elements, image_summaries = get_image_summaries(openai_api_key, output_path)
        
        documents = []
        retrieve_contents = []

        for e, s in zip(text_elements, text_summaries):
            i = str(uuid.uuid4())
            doc = Document(
                page_content=s, metadata={"id": i, "type": "text", "original_content": e}
            )
            retrieve_contents.append((i, e))
            documents.append(doc)

        for e, s in zip(table_elements, table_summaries):
            i = str(uuid.uuid4())
            doc = Document(
                page_content=s, metadata={"id": i, "type": "table", "original_content": e}
            )
            retrieve_contents.append((i, e))
            documents.append(doc)

        for e, s in zip(image_elements, image_summaries):
            i = str(uuid.uuid4())
            doc = Document(
                page_content=s, metadata={"id": i, "type": "image", "original_content": e}
            )
            retrieve_contents.append((i, s))
            documents.append(doc)

        vectorstore = FAISS.from_documents(
            documents=documents, embedding=HuggingFaceBgeEmbeddings()
        )
        logger.info("Vectorstore created successfully.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating document and vectorstore: {e}")
        raise

prompt_template = """
You are an expert in historical monuments, their architectural styles, cultural significance, and related details.
Answer the question based only on the following context, which can include text, images, and tables:
{context}
Question: {question}
You will only reponse those question which answers present in context. if not in context then say "Sorry, I don't have much information about it."
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return a helpful answer with as much detail as possible.
Answer:
"""

def get_response_from_llm(vectorstore, question, openai_api_key):
    logger.info("Retrieving response from LLM.")
    try:
        qa_chain = LLMChain(
            llm=ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key, max_tokens=1024),
            prompt=PromptTemplate.from_template(prompt_template),
        )
        relevant_docs = vectorstore.similarity_search(question)
        context = ""
        relevant_images = []
        for d in relevant_docs:
            if d.metadata["type"] == "text":
                context += "[text]" + d.metadata["original_content"]
            elif d.metadata["type"] == "table":
                context += "[table]" + d.metadata["original_content"]
            elif d.metadata["type"] == "image":
                context += "[image]" + d.page_content
                relevant_images.append(d.metadata["original_content"])
        result = qa_chain.run({"context": context, "question": question})
        logger.info("Response retrieved successfully.")
        return result, relevant_images
    except Exception as e:
        logger.error(f"Error retrieving response from LLM: {e}")
        raise


if __name__ == "__main__":
    pdf_path = "mouments-1-5.pdf"
    output_path = "uploads/"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    vectorstore = create_document_and_vectorstore(openai_api_key, pdf_path, output_path)
    question = "Tell me about Taj Mahal."
    result , relevant_images= get_response_from_llm(vectorstore, question, openai_api_key)
    print(result)
    print(relevant_images,777777777777777777777777)
