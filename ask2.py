# ask.py
# Hybrid Q&A:
# - Quantity questions -> SQL on public.products
# - Everything else -> RAG via PGVectorStore (lc_rag_chunks) + Gemini generate

import os
import re
import asyncio
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

from google import genai
from google.genai.errors import ClientError

from langchain_postgres import PGEngine, PGVectorStore

from embeddings_gemini import GeminiEmbeddings  # ต้องมีไฟล์นี้ในโปรเจกต์ (ที่เคยให้ไป)


# -----------------------------
# Config
# -----------------------------
VECTOR_TABLE = os.getenv("VECTOR_TABLE", "lc_rag_chunks")   # ตาราง vector store ของ LangChain
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "1536"))         # ต้องตรงกับ output_dimensionality
TOP_K = int(os.getenv("TOP_K", "4"))

# โมเดลตอบ (ปรับได้ใน .env)
GEN_MODEL = os.getenv("GEN_MODEL", "gemini-2.5-flash")

SYSTEM_PROMPT = """คุณคือผู้ช่วยตอบคำถาม
- ถ้าตอบจาก Context ได้ ให้ตอบโดยอ้างอิงจาก Context เท่านั้น
- ถ้า Context ไม่พอ ให้ตอบว่า "ไม่พบข้อมูลในเอกสาร"
- ตอบเป็นภาษาไทย
- ถ้าอ้างอิง ให้ใส่เลขแหล่งที่มา [1], [2] ต่อท้ายประโยค
"""


# -----------------------------
# Quantity intent detection
# -----------------------------
QTY_KEYWORDS = [
    "สต็อก", "stock", "คงเหลือ", "เหลือ", "จำนวน", "ปริมาณ",
    "มีเท่าไหร่", "กี่ชิ้น", "กี่อัน", "ทั้งหมด", "รวม", "นับ", "กี่รายการ",
    "เหลืออยู่", "เหลือกี่", "รวมกี่", "รวมทั้งหมด",
    "ต่ำกว่า", "น้อยกว่า"
]

def is_quantity_question(q: str) -> bool:
    ql = q.lower()
    if any(k in ql for k in QTY_KEYWORDS):
        return True
    # ถ้ามี pattern ตัวเลขที่มักเกี่ยวกับปริมาณ
    if re.search(r"(?:ต่ำกว่า|น้อยกว่า)\s*\d+", ql):
        return True
    return False


def extract_product_id(q: str) -> Optional[str]:
    # รองรับ P001 หรือ p001
    m = re.search(r"\b(p\d{1,6})\b", q, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def extract_name_hint(q: str) -> str:
    # heuristics ง่าย ๆ เพื่อเอาคำสำคัญไปหา name
    # ตัดคำทั่วไปออก
    stop = set(["สต็อก", "เหลือ", "คงเหลือ", "เท่าไหร่", "กี่", "ชิ้น", "อัน", "ราคา", "รวม", "ทั้งหมด", "ตอนนี้", "มี", "ไหม"])
    tokens = [t for t in re.split(r"\s+", q.strip()) if t and t not in stop]
    # เอา 1-3 token แรกมารวมเป็น hint
    return " ".join(tokens[:3]).strip()


# -----------------------------
# SQL Answering (public.products)
# -----------------------------
SQL_COUNT_PRODUCTS = "select count(*) from public.products;"
SQL_SUM_STOCK = "select coalesce(sum(stock), 0) from public.products;"

SQL_LOW_STOCK = """
select product_id, name, stock
from public.products
where stock < :x
order by stock asc, name asc
limit 20;
"""

SQL_BY_ID = """
select product_id, name, stock, price_thb
from public.products
where product_id = :pid
limit 1;
"""

SQL_BY_NAME_ILIKE = """
select product_id, name, stock, price_thb
from public.products
where name ilike :pat
order by length(name) asc
limit 5;
"""


async def answer_from_db(conn, question: str) -> Optional[str]:
    q = question.strip()

    # 1) นับจำนวนสินค้า
    if any(k in q for k in ["กี่สินค้า", "กี่รายการ", "มีกี่สินค้า", "มีกี่รายการ", "นับสินค้า"]):
        r = await conn.execute(text(SQL_COUNT_PRODUCTS))
        n = r.scalar() or 0
        return f"ตอนนี้มีสินค้า {n} รายการในระบบ"

    # 2) รวมสต็อกทั้งหมด
    if any(k in q for k in ["รวมสต็อก", "สต็อกรวม", "รวมทั้งหมด", "ทั้งหมดมีเท่าไหร่"]):
        r = await conn.execute(text(SQL_SUM_STOCK))
        total = r.scalar() or 0
        return f"สต็อกรวมทั้งหมดตอนนี้คือ {int(total)} ชิ้น"

    # 3) สต็อกต่ำกว่า X
    m = re.search(r"(ต่ำกว่า|น้อยกว่า)\s*(\d+)", q)
    if m:
        x = int(m.group(2))
        r = await conn.execute(text(SQL_LOW_STOCK), {"x": x})
        rows = r.fetchall()
        if not rows:
            return f"ไม่มีสินค้าที่สต็อกต่ำกว่า {x} ชิ้น"
        lines = [f"- {row.name} (เหลือ {row.stock})" for row in rows]
        return f"สินค้าที่สต็อกต่ำกว่า {x} ชิ้น:\n" + "\n".join(lines)

    # 4) ถามสต็อกของสินค้าเจาะจง: ถ้ามี product_id
    pid = extract_product_id(q)
    if pid:
        r = await conn.execute(text(SQL_BY_ID), {"pid": pid})
        row = r.fetchone()
        if row:
            price = row.price_thb
            return f"{row.name} (ID {row.product_id}) เหลือ {row.stock} ชิ้น ราคา {price} บาท"
        return f"ไม่พบสินค้า ID {pid} ในระบบ"

    # 5) ถามสต็อกโดยใช้ชื่อ (ILIKE)
    # เฉพาะเมื่อในคำถามมีความหมายเกี่ยวกับจำนวน/เหลือ
    if any(k in q for k in ["สต็อก", "เหลือ", "คงเหลือ", "จำนวน", "กี่ชิ้น", "มีเท่าไหร่"]):
        hint = extract_name_hint(q)
        if not hint:
            hint = q

        r = await conn.execute(text(SQL_BY_NAME_ILIKE), {"pat": f"%{hint}%"})
        rows = r.fetchall()

        if not rows and hint != q:
            # fallback: ใช้คำถามเต็ม
            r = await conn.execute(text(SQL_BY_NAME_ILIKE), {"pat": f"%{q}%"})
            rows = r.fetchall()

        if rows:
            if len(rows) == 1:
                row = rows[0]
                return f"{row.name} (ID {row.product_id}) เหลือ {row.stock} ชิ้น ราคา {row.price_thb} บาท"
            else:
                lines = [f"- {row.name} (ID {row.product_id}) เหลือ {row.stock}" for row in rows]
                return "เจอหลายรายการที่คล้ายกัน:\n" + "\n".join(lines) + "\nพิมพ์ชื่อให้ชัดขึ้นหรือใส่ product_id (เช่น P102) ได้ไหม?"

    return None  # DB ตอบไม่ได้ -> ไป RAG


# -----------------------------
# RAG formatting
# -----------------------------
def format_context(docs) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        extra = []
        if "page" in d.metadata:
            extra.append(f"page={d.metadata['page']}")
        if "row" in d.metadata:
            extra.append(f"row={d.metadata['row']}")
        if "product_id" in d.metadata:
            extra.append(f"product_id={d.metadata['product_id']}")
        extra_txt = (" " + ",".join(extra)) if extra else ""
        parts.append(f"[{i}] source={src}{extra_txt}\n{d.page_content}")
    return "\n\n".join(parts)


def format_sources(docs) -> str:
    lines = []
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.metadata}")
    return "\n".join(lines)


# -----------------------------
# Gemini generate with retry (กัน rate-limit แบบชั่วคราว)
# -----------------------------
def gemini_generate(client: genai.Client, prompt: str, model: str) -> str:
    # retry แบบเบา ๆ (ถ้า quota = 0 จะยังพังอยู่ แต่จะบอกข้อความชัด)
    for attempt in range(3):
        try:
            resp = client.models.generate_content(model=model, contents=prompt)
            return (resp.text or "").strip()
        except ClientError as e:
            # 429 = rate limit / quota
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == 2:
                    raise
                # backoff ง่าย ๆ
                import time
                time.sleep(1.5 * (attempt + 1))
                continue
            raise
    return ""


# -----------------------------
# Main
# -----------------------------
async def main():
    load_dotenv()

    db_url = os.environ.get("DATABASE_URL")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not db_url:
        raise SystemExit("Missing DATABASE_URL in .env")
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY in .env")

    # DB engine (สำหรับถามปริมาณ)
    db_engine = create_async_engine(db_url, pool_pre_ping=True)

    # Vector store (สำหรับ RAG)
    embedding = GeminiEmbeddings(api_key=api_key, dims=VECTOR_SIZE)
    pg_engine = PGEngine.from_connection_string(url=db_url)
    store = await PGVectorStore.create(
        engine=pg_engine,
        table_name=VECTOR_TABLE,
        embedding_service=embedding,
    )

    # Gemini client (ตอบ)
    gem_client = genai.Client(api_key=api_key)

    print("✅ Ready. Type your question (type 'exit' to quit).")

    while True:
        q = input("\nQ: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        # 1) ถ้าเป็นคำถามปริมาณ -> ตอบจาก DB ตรง
        if is_quantity_question(q):
            try:
                async with db_engine.connect() as conn:
                    ans = await answer_from_db(conn, q)
                if ans:
                    print("\nA:", ans)
                    continue
            except Exception as e:
                # ถ้า DB ตอบพัง ก็ fallback ไป RAG
                print("\n⚠️ DB query failed, fallback to RAG:", type(e).__name__, str(e))

        # 2) RAG fallback
        try:
            docs = await store.asimilarity_search(q, k=TOP_K)
        except Exception as e:
            print("\n❌ Vector search failed:", type(e).__name__, str(e))
            continue

        context = format_context(docs)
        prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {q}"

        try:
            answer = gemini_generate(gem_client, prompt, GEN_MODEL)
        except ClientError as e:
            msg = str(e)
            # เคส quota = 0
            if "RESOURCE_EXHAUSTED" in msg and "limit: 0" in msg:
                print("\n❌ Gemini quota for this key/project is 0 (free-tier limit is 0). ต้องเปิดใช้งาน/ผูก billing หรือใช้ key ที่มี quota.")
            else:
                print("\n❌ Gemini error:", msg)
            continue
        except Exception as e:
            print("\n❌ Gemini error:", type(e).__name__, str(e))
            continue

        print("\nA:", answer)

        if docs:
            print("\n--- Sources ---")
            print(format_sources(docs))

    await db_engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
