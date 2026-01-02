from __future__ import annotations
import os
import glob
import pandas as pd
from pypdf import PdfReader
from langchain_core.documents import Document


def load_txt_md(folder: str) -> list[Document]:
    docs: list[Document] = []
    patterns = ["**/*.txt", "**/*.md"]
    for pat in patterns:
        for path in glob.glob(os.path.join(folder, pat), recursive=True):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": path}))
    return docs


def load_pdf(folder: str) -> list[Document]:
    docs: list[Document] = []
    for path in glob.glob(os.path.join(folder, "**/*.pdf"), recursive=True):
        reader = PdfReader(path)
        for page_idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if not text:
                # สแกนรูป/ไม่มี text จริง -> ข้าม (หรือคุณจะเก็บแจ้งเตือนไว้ก็ได้)
                continue
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "page": page_idx}
            ))
    return docs


def load_csv(folder: str, text_cols: list[str] | None = None, max_rows: int | None = None) -> list[Document]:
    docs: list[Document] = []
    for path in glob.glob(os.path.join(folder, "**/*.csv"), recursive=True):
        # ลองอ่านแบบปกติก่อน ถ้าพัง ค่อย fallback
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(
                path,
                engine="python",
                sep=None,              # auto-detect delimiter
                on_bad_lines="skip",   # ข้ามบรรทัดที่พัง
                encoding_errors="ignore",
            )

        if max_rows:
            df = df.head(max_rows)

        cols = text_cols or list(df.columns)

        for i, row in df.iterrows():
            parts = []
            for c in cols:
                val = row.get(c, "")
                if pd.isna(val):
                    val = ""
                parts.append(f"{c}: {val}")
            text = "\n".join(parts).strip()
            if not text:
                continue

            docs.append(Document(
                page_content=text,
                metadata={"source": path, "row": int(i)}
            ))
    return docs


def load_all(folder: str, csv_cols: list[str] | None = None) -> list[Document]:
    docs = []
    docs += load_txt_md(folder)
    docs += load_pdf(folder)
    docs += load_csv(folder, text_cols=csv_cols)
    return docs
