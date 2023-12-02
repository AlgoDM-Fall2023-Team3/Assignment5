import streamlit as st
import os
import time
import random, string
from pathlib import Path
from PIL import Image
import qdrant_client
from llama_index import SimpleDirectoryReader
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index import StorageContext
from llama_index.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.response.notebook_utils import display_source_node
from llama_index.schema import ImageNode
from llama_index.multi_modal_llms.openai import OpenAIMultiModal
from llama_index.schema import ImageDocument

# Access API key from Streamlit secrets
OPEN_API_KEY = st.secrets['open_api_key']
os.environ["OPENAI_API_KEY"] = OPEN_API_KEY
image_path = './FashionImages'

def create_llm_index(client):
    text_store = QdrantVectorStore(
        client=client, collection_name="text_collection"
    )
    image_store = QdrantVectorStore(
        client=client, collection_name="image_collection"
    )
    storage_context = StorageContext.from_defaults(vector_store=text_store)

    documents = SimpleDirectoryReader("./FashionImages").load_data()
    index = MultiModalVectorStoreIndex.from_documents(
        documents, storage_context=storage_context, image_vector_store=image_store
    )

    return index

def retrieve_images(query, index):
    # Create Llama index

    retriever = index.as_retriever(similarity_top_k=1, image_similarity_top_k=1)
    retrieval_results = retriever.retrieve(query)

    retrieved_images = []
    for res_node in retrieval_results:
        if isinstance(res_node.node, ImageNode):
            retrieved_images.append(res_node.node.metadata["file_path"])
    return retrieved_images

def image_retrieval(input_image_path, index):
    

    retriever_engine = index.as_retriever(image_similarity_top_k=2)
    # retrieve more information from the GPT4V response

    retrieval_results = retriever_engine.image_to_image_retrieve(
        "./FashionImages/"+input_image_path
    )
    retrieved_images = []
    image_documents = [ImageDocument(image_path="./FashionImages/"+input_image_path)]

    for res in retrieval_results:
        retrieved_images.append(res.node.metadata["file_path"])

    for res_img in retrieved_images[1:]:
        image_documents.append(ImageDocument(image_path=res_img))
    
    openai_mm_llm = OpenAIMultiModal(
        model="gpt-4-vision-preview", api_key=os.environ["OPENAI_API_KEY"], max_new_tokens=1500
    )
    response = openai_mm_llm.complete(
        prompt="Given the first image as the base image, what the other images correspond to?",
        image_documents=image_documents,
    )
    
    return retrieved_images, response

# Streamlit app
def main():
    st.title("Multi-Modal Image Retrieval System")

    c1, c2 = st.tabs(["Search by text", "Search by Image"])
    client = qdrant_client.QdrantClient(":memory:")
    index = create_llm_index(client)

    with c1:
        # Get user input query
        query = st.text_input("Enter a description to find matching images:")
       

        if st.button("Search"):
            # Retrieve images based on the query
            retrieved_images = retrieve_images(query, index)

            # Display retrieved images
            if retrieved_images:
                st.subheader("Matching Images:")
                for img_path in retrieved_images:
                    st.image(img_path, caption=os.path.basename(img_path), use_column_width=True)
            else:
                st.warning("No matching images found.")

    with c2:
        # Upload an image
        st.title("Image Retrieval and Reasoning App")

        # Upload input image
        input_image = st.file_uploader("Upload Input Image", type=["jpg", "png"])
        if input_image:
            st.image(input_image, caption="Uploaded Input Image.", use_column_width=True)
            input_image_path = input_image.name

            # Retrieve images based on the input image
            retrieved_images_paths = image_retrieval(input_image_path, index)[0]

            # Display retrieved images
            st.subheader("Retrieved Images")
            for img_path in retrieved_images_paths:
                st.image(img_path, caption=img_path, use_column_width=True)

            # Display GPT-4V reasoning response
            st.subheader("GPT-4V Reasoning")
            reasoning_response = image_retrieval(input_image_path, index)[1]
            st.write(reasoning_response)

if __name__ == "__main__":
    main()
