import os
import asyncio
import hashlib
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_postgres import PGEngine, PGVectorStore

from embeddings_gemini import GeminiEmbeddings  # ไฟล์ที่คุณมีอยู่แล้ว (ล็อก dims)

VECTOR_SIZE = 1536
VECTOR_TABLE = "lc_rag_chunks"        # ตาราง vector store
SOURCE_TAG = "supabase:products"      # เอาไว้ลบ/กันซ้ำ


def row_to_text(row: dict) -> str:
    # ทำให้แต่ละแถวเป็น “ข้อความค้นง่าย”
    # ปรับคอลัมน์ได้ตามจริงของตารางคุณ
    return "\n".join([
        f"product_id: {row.get('product_id','')}",
        f"name: {row.get('name','')}",
        f"category: {row.get('category','')}",
        f"price_thb: {row.get('price_thb','')}",
        f"stock: {row.get('stock','')}",
        f"description: {row.get('description','')}",
        f"tags: {row.get('tags','')}",
        f"updated_at: {row.get('updated_at','')}",
    ]).strip()


def content_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


async def fetch_products(db_url: str) -> list[dict]:
    engine = create_async_engine(db_url, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            res = await conn.execute(text("""
                select product_id, name, category, price_thb, stock, description, tags, updated_at
                from public.products
                order by product_id asc;
            """))
            rows = [dict(r._mapping) for r in res.fetchall()]
            return rows
    finally:
        await engine.dispose()


async def purge_old_vectors(db_url: str):
    # ลบเฉพาะสิ่งที่ ingest จาก source นี้ (กันข้อมูลซ้ำ)
    engine = create_async_engine(db_url, pool_pre_ping=True)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(f"delete from public.{VECTOR_TABLE} where langchain_metadata->>'source' = :src"),
                {"src": SOURCE_TAG},
            )
    finally:
        await engine.dispose()


async def main():
    load_dotenv()
    db_url = os.environ["DATABASE_URL"]
    api_key = os.environ["GEMINI_API_KEY"]

    print("1) Fetching from Supabase tables...")
    rows = await fetch_products(db_url)
    if not rows:
        print("⚠️ No rows in public.products")
        return

    docs: list[Document] = []
    for r in rows:
        txt = row_to_text(r)
        docs.append(Document(
            page_content=txt,
            metadata={
                "source": SOURCE_TAG,
                "product_id": r.get("product_id"),
                "hash": content_hash(txt),
                "updated_at": str(r.get("updated_at") or ""),
            }
        ))

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    print("2) Purging old vectors for this source...")
    await purge_old_vectors(db_url)

    print("3) Writing vectors...")
    embedding = GeminiEmbeddings(api_key=api_key, dims=VECTOR_SIZE)

    pg_engine = PGEngine.from_connection_string(url=db_url)
    # ถ้ายังไม่เคยสร้าง vector table ให้เปิดบรรทัดนี้ (ครั้งแรกครั้งเดียว)
    # await pg_engine.ainit_vectorstore_table(table_name=VECTOR_TABLE, vector_size=VECTOR_SIZE)

    store = await PGVectorStore.create(
        engine=pg_engine,
        table_name=VECTOR_TABLE,
        embedding_service=embedding,
    )

    await store.aadd_documents(chunks)
    print(f"✅ Ingested {len(chunks)} chunks from Supabase DB into {VECTOR_TABLE}")


if __name__ == "__main__":
    asyncio.run(main())
