from typing import List

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_unstructured import UnstructuredLoader
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import time


load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

urls = [
    #articles about generative AI / autonomous AI
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    #prompt Engineering article
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    #article on adverserial attack on llms
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",

]
docs=list()
for url in urls:
        docs.extend(UnstructuredLoader(web_url=url ,     chunking_strategy="basic",max_characters =1000000).load())


# print(docs)


text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size =250,
    chunk_overlap =0
)

#docList
doc_split  = text_splitter.split_documents(docs)
#
vectorstore = Chroma(
    collection_name="rag-chroma",
    embedding_function=embeddings,
    persist_directory="./.chroma",
)
# batch_size=20
# need batches cuase got no gemini money :_(

# for i in range(0, len(doc_split), batch_size):
#     batch = doc_split[i:i + batch_size]
#
#     vectorstore.add_documents(batch)
#
#     print(f"Embedded {min(i + batch_size, len(doc_split))}/{len(doc_split)} chunks")
#
#     # Wait 15 seconds before the next batch
#     if i + batch_size < len(doc_split):
#         time.sleep(15)


retriever = vectorstore.as_retriever()
