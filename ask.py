import os
import asyncio
from dotenv import load_dotenv
from google import genai

from langchain_postgres import PGEngine, PGVectorStore
from embeddings_gemini import GeminiEmbeddings

TABLE_NAME = "lc_rag_chunks"
VECTOR_SIZE = 1536

SYSTEM = """ตอบโดยอ้างอิงจาก Context เท่านั้น
ถ้า Context ไม่พอ ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร"
ตอบเป็นภาษาไทย
"""

def format_context(docs):
    out = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        extra = []
        if "page" in d.metadata: extra.append(f"page={d.metadata['page']}")
        if "row" in d.metadata: extra.append(f"row={d.metadata['row']}")
        extra_txt = (" " + ",".join(extra)) if extra else ""
        out.append(f"[{i}] {src}{extra_txt}\n{d.page_content}")
    return "\n\n".join(out)

async def main():
    load_dotenv()
    api_key = os.environ["GEMINI_API_KEY"]
    db_url = os.environ["DATABASE_URL"]

    embedding = GeminiEmbeddings(api_key=api_key, dims=VECTOR_SIZE)
    pg_engine = PGEngine.from_connection_string(url=db_url)

    store = await PGVectorStore.create(
        engine=pg_engine,
        table_name=TABLE_NAME,
        embedding_service=embedding,
    )

    q = input("Q: ").strip()
    docs = await store.asimilarity_search(q, k=4)

    context = format_context(docs)
    prompt = f"{SYSTEM}\n\nContext:\n{context}\n\nQuestion: {q}"

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    print("\n--- Answer ---")
    print(resp.text)

    print("\n--- Sources ---")
    for d in docs:
        print("-", d.metadata.get("source"), d.metadata)

if __name__ == "__main__":
    asyncio.run(main())
