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
from concurrent.futures import ThreadPoolExecutor , as_completed
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_raw_pdf_elements(pdf_path, output_path,my_bar):
    logger.info(f"Extracting elements from PDF: {pdf_path}")
    my_bar.progress(20,text="Extracting elements from PDF")
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


def summarize_single_text(element, summary_chain):
    """Helper function to summarize a single text element."""
    return summary_chain.run({"element_type": "text", "element": element})

def summarize_text(openai_api_key, raw_pdf_elements, my_bar):
    logger.info("Starting text and table summarization.")
    my_bar.progress(40, text="Starting text and table summarization")
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

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for e in raw_pdf_elements:
                if "CompositeElement" in repr(e):
                    logger.debug("Summarizing a text element.")
                    text_elements.append(e.text)
                    futures.append(executor.submit(summarize_single_text, e, summary_chain))

                elif "Table" in repr(e):
                    logger.debug("Summarizing a table element.")
                    table_elements.append(e.text)
                    futures.append(executor.submit(summarize_single_text, e, summary_chain))

            # Collect results
            for future in futures:
                summary = future.result()  # Wait for the result
                if summary:
                    text_summaries.append(summary)


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

def summarize_single_image(image_path, openai_api_key, my_bar=None):

    try:
        logger.info(f"Processing image: {image_path}")
        
        # Encode the image
        encoded_image_ = encode_image(image_path)
        
        # Prepare the prompt for image summarization
        prompt = [
            SystemMessage(content="You are a bot that is good at analyzing images."),
            HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the contents of this image in detail."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded_image_}"},
                    },
                ]
            ),
        ]
        
        # Use OpenAI Vision API to summarize the image
        response = ChatOpenAI(
            model="gpt-4-vision-preview", 
            openai_api_key=openai_api_key, 
            max_tokens=1024
        ).invoke(prompt)
        
        logger.info(f"Successfully summarized image: {response.content}'\n")
        return encoded_image_, response.content
    
    except Exception as e:
        logger.error(f"Error summarizing image {image_path}: {e}")
        return image_path, None

def get_image_summaries(openai_api_key, output_path, my_bar=None, max_workers=3):

    logger.info("Gathering image summaries.")
    
    # Filter image files
    image_files = [
        os.path.join(output_path, i) 
        for i in os.listdir(output_path) 
        if i.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    
    image_elements = []
    image_summaries = []
    
    # Use ThreadPoolExecutor with a limited number of workers
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all image summarization tasks
        future_to_image = {
            executor.submit(summarize_single_image, image_path, openai_api_key, my_bar): image_path 
            for image_path in image_files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_image):
            image_path = future_to_image[future]
            try:
                # Get the result of the task
                encode_image, summary = future.result()

                
                # Only add successful summaries
                if summary:
                    image_elements.append(encode_image)
                    image_summaries.append(summary)
            
            except Exception as e:
                logger.error(f"Error processing future for {image_path}: {e}")
    
    logger.info(f"Image summaries collected: {len(image_summaries)} out of {len(image_files)}")
    return image_elements, image_summaries


def create_document_and_vectorstore(openai_api_key, pdf_path, output_path,my_bar):
    logger.info(f"Creating document and vectorstore for PDF: {pdf_path}")
    my_bar.progress(80, text="Creating document and vectorstore")
    try:
        raw_pdf_elements = get_raw_pdf_elements(pdf_path, output_path,my_bar)
        text_elements, text_summaries, table_elements, table_summaries = summarize_text(
            openai_api_key, raw_pdf_elements,my_bar
        )
        image_elements, image_summaries = get_image_summaries(openai_api_key, output_path,my_bar)
        
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
Answer all the question to the user based only on the following context, which can include text, images, and tables:
{context}
Question: {question}
Memory: {memory}
Always use the user's memory while giving the answer, as the user's chat history is saved. If the user asks about a previous question, give them the correct answer based on the memory.
You will only reponse those question which answers present in context. if not in context then say "Sorry, I don't have much information about it."
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
You will always respond user greetings.
Just return a helpful answer with as much detail as possible.
Answer:
"""

def get_response_from_llm(vectorstore, question, openai_api_key,memory):
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
        result = qa_chain.run({"context": context, "question": question,"memory":memory})
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
