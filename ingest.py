import os
import asyncio
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGEngine, PGVectorStore

from embeddings_gemini import GeminiEmbeddings
from loaders import load_all

TABLE_NAME = "lc_rag_chunks"
VECTOR_SIZE = 1536

async def main():
    load_dotenv()
    api_key = os.environ["GEMINI_API_KEY"]
    db_url = os.environ["DATABASE_URL"]

    embedding = GeminiEmbeddings(api_key=api_key, dims=VECTOR_SIZE)

    pg_engine = PGEngine.from_connection_string(url=db_url)

    # สร้างตาราง schema ให้ถูกต้อง (ถ้าคุณสร้างเองแล้วก็ไม่เป็นไร)
    await pg_engine.ainit_vectorstore_table(table_name=TABLE_NAME, vector_size=VECTOR_SIZE)

    store = await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_NAME,
        embedding_service=embedding,
    )

    # เลือกคอลัมน์ CSV สำคัญ (ถ้าเป็นสินค้าร้าน แนะนำกำหนดให้ชัด)
    docs = load_all("data", csv_cols=None)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    await store.aadd_documents(chunks)

    print(f"✅ Ingested {len(chunks)} chunks into {TABLE_NAME}")

if __name__ == "__main__":
    asyncio.run(main())
