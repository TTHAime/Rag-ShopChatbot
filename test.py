import os
import asyncio
from dotenv import load_dotenv

from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from google import genai
from google.genai import types


async def test_db(db_url: str):
    print("== DB TEST ==")
    engine = create_async_engine(db_url, pool_pre_ping=True)

    try:
        async with engine.connect() as conn:
            # 1) basic query
            r = await conn.execute(text("select 1 as ok;"))
            print("DB select 1:", r.scalar())

            # 2) check pgvector extension
            r = await conn.execute(text("select extname from pg_extension where extname='vector';"))
            ext = r.scalar()
            if ext == "vector":
                print("pgvector extension: OK")
            else:
                print("pgvector extension: NOT FOUND (ต้อง create extension vector ก่อน)")

    finally:
        await engine.dispose()


def test_gemini(api_key: str, dims: int = 1536):
    print("\n== GEMINI TEST ==")
    client = genai.Client(api_key=api_key)

    # 1) generate test
    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Reply with exactly: OK",
    )
    print("Gemini generate:", resp.text.strip())

    # 2) embedding test
    emb = client.models.embed_content(
        model="gemini-embedding-001",
        contents="hello world",
        config=types.EmbedContentConfig(output_dimensionality=dims),
    )
    vec_len = len(emb.embeddings[0].values)
    print(f"Gemini embedding: OK (dims={vec_len})")


async def main():
    load_dotenv()
    db_url = os.getenv("DATABASE_URL")
    api_key = os.getenv("GEMINI_API_KEY")

    if not db_url:
        raise SystemExit("Missing DATABASE_URL in .env")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY in .env")

    # DB
    try:
        await test_db(db_url)
    except Exception as e:
        print("DB ERROR:", type(e).__name__, e)

    # Gemini
    try:
        test_gemini(api_key, dims=1536)
    except Exception as e:
        print("GEMINI ERROR:", type(e).__name__, e)


if __name__ == "__main__":
    asyncio.run(main())
