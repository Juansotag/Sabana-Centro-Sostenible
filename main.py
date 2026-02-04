"""
Sabana Centro Sostenible — Extractor de referencias (Planes de Desarrollo -> Proyectos)

Este script:
1) Lee `Proyectos.xlsx` (en la raíz).
2) Lee PDFs en `./documentos/` con convención: MUNICIPIO-ALGO.pdf (mayúsculas, sin tildes).
3) Extrae texto por páginas, lo fragmenta (chunks) y mantiene trazabilidad (página inicio/fin).
4) Usa embeddings para seleccionar chunks candidatos por proyecto.
5) Usa un LLM para validar cada candidato y generar:
   - justificación breve
   - resumen corto del fragmento
6) Exporta resultados a: `./outputs/referencias/referencias_generadas.xlsx`

Notas importantes:
- Variables de entorno (recomendado por .env):
    OPENAI_API_KEY=...
    SIM_THRESHOLD=0.35
    TOP_K=8
  (Opcionales: OPENAI_EMBED_MODEL, OPENAI_LLM_MODEL, CHUNK_CHARS, CHUNK_OVERLAP)

- Lectura de PDF requiere instalar UNO:
    pip install pymupdf
  o:
    pip install pdfplumber

- Barra de progreso:
    pip install tqdm
  Si no está, igual corre sin barra.

Compatibilidad OpenAI SDK:
- NO usamos `text.format` ni `json_schema` (evita el error 400: Missing required parameter 'text.format.name').
- Forzamos salida JSON desde el prompt y parseamos con json.loads.
"""

from __future__ import annotations

import os
import re
import json
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv


# -------------------------
# Carga variables desde .env
# -------------------------
load_dotenv()  # Carga variables desde un archivo .env en la raíz del proyecto (NO lo subas a Git)


# -------------------------
# Paths (estructura simplificada)
# -------------------------
BASE_DIR = Path(".")
PROJECTS_XLSX = BASE_DIR / "Proyectos.xlsx"
DOCUMENTS_DIR = BASE_DIR / "documentos"
OUTPUT_DIR = BASE_DIR / "outputs" / "referencias"
OUTPUT_XLSX = OUTPUT_DIR / "referencias_generadas.xlsx"


# -------------------------
# Config OpenAI
# -------------------------
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")


# -------------------------
# Parámetros de búsqueda / extracción
# -------------------------
TOP_K_PER_PROJECT_PER_DOC = int(os.getenv("TOP_K", "12"))
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.30"))

CHUNK_CHARS = int(os.getenv("CHUNK_CHARS", "1800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "250"))

EXTRACTOR_VERSION = "v0.1.2"


# -------------------------
# Municipios (normalizados sin tildes)
# -------------------------
MUNICIPIOS_SABANA_CENTRO = {
    "CAJICA", "CHIA", "COGUA", "COTA", "GACHANCIPA", "NEMOCON",
    "SOPO", "TABIO", "TENJO", "TOCANCIPA", "ZIPAQUIRA",
}


# -------------------------
# Barra de progreso (tqdm opcional)
# -------------------------
def progress(iterable, total: Optional[int] = None, desc: str = ""):
    """
    Devuelve un iterable con barra de progreso si `tqdm` está instalado;
    si no, devuelve el iterable normal.
    """
    try:
        from tqdm import tqdm  # type: ignore
        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


# -------------------------
# Utilidades: normalización
# -------------------------
def normalize(s: str) -> str:
    """Pasa a mayúsculas y quita tildes/diacríticos."""
    import unicodedata
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"\s+", " ", s.strip())
    return s.upper()


def infer_municipio_from_filename(filename: str) -> str:
    """
    Intenta inferir municipio desde el prefijo del nombre:
      SOPO-PLAN-DE-DESARROLLO.pdf -> SOPO
    """
    base = Path(filename).stem
    parts = base.split("-")
    guess = normalize(parts[0]) if parts else "DESCONOCIDO"

    if guess in MUNICIPIOS_SABANA_CENTRO:
        return guess

    n = normalize(base)
    for m in MUNICIPIOS_SABANA_CENTRO:
        if m in n:
            return m

    return guess


# -------------------------
# Extracción de texto desde PDF
# -------------------------
def extract_pages_text(pdf_path: Path) -> List[Tuple[int, str]]:
    """
    Devuelve [(numero_pagina_1based, texto), ...].

    Intenta PyMuPDF (fitz). Si no está disponible, intenta pdfplumber.
    """
    # Intento 1: PyMuPDF
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            text = doc[i].get_text("text") or ""
            pages.append((i + 1, text))
        return pages
    except Exception:
        pass

    # Intento 2: pdfplumber
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                pages.append((i + 1, text))
        return pages
    except Exception as e:
        raise RuntimeError(
            f"No pude extraer texto de {pdf_path}. "
            f"Instala PyMuPDF (pymupdf) o pdfplumber. Error: {e}"
        )


def chunk_text(pages: List[Tuple[int, str]], chunk_chars: int, overlap: int) -> List[Dict]:
    """
    Crea fragmentos (chunks) por caracteres conservando trazabilidad por páginas.
    """
    full = []
    page_spans = []  # (start_idx, end_idx, page_no)
    cursor = 0

    for page_no, text in pages:
        t = (text or "").strip()
        if not t:
            continue
        full.append(t + "\n")
        start = cursor
        cursor += len(t) + 1
        end = cursor
        page_spans.append((start, end, page_no))

    big = "".join(full)
    if not big.strip():
        return []

    chunks = []
    step = max(1, chunk_chars - overlap)

    for start in range(0, len(big), step):
        end = min(len(big), start + chunk_chars)
        chunk = big[start:end].strip()

        # Evita fragmentos demasiado cortos (ruido)
        if len(chunk) < 150:
            if end == len(big):
                break
            continue

        covered_pages = [p for (s, e, p) in page_spans if not (e <= start or s >= end)]
        if not covered_pages:
            if end == len(big):
                break
            continue

        chunks.append({
            "chunk_text": chunk,
            "page_start": min(covered_pages),
            "page_end": max(covered_pages),
        })

        if end == len(big):
            break

    return chunks


# -------------------------
# OpenAI: cliente y utilidades
# -------------------------
def get_openai_client():
    """
    Crea el cliente OpenAI usando OPENAI_API_KEY.
    La key debe estar en el .env o como variable de entorno del sistema.
    """
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Falta OPENAI_API_KEY en variables de entorno (revisa tu .env).")
    return OpenAI(api_key=api_key)


def embed_texts(client, texts: List[str]) -> List[List[float]]:
    """Embeddings en batch."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def cosine(a: List[float], b: List[float]) -> float:
    """Similitud coseno entre vectores."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def llm_filter_and_justify(
    client,
    proyecto: Dict,
    municipio: str,
    documento: str,
    chunk: Dict
) -> Optional[Dict]:
    """
    Filtro final con LLM para reducir falsos positivos.

    Importante:
    - NO usamos `text.format` ni JSON schema (evita errores por versión del SDK).
    - Forzamos salida JSON desde el prompt y luego hacemos json.loads.
    """
    sys = (
        "Eres un analista técnico de planeación territorial y sostenibilidad. "
        "Tu tarea es validar si un fragmento de un Plan de Desarrollo municipal "
        "aporta (directa o indirectamente) a un proyecto estratégico de Sabana Centro Sostenible. "
        "Sé conservador: si el fragmento no evidencia relación clara con el objetivo o requerimientos, responde NO."
    )

    payload = {
        "municipio": municipio,
        "documento": documento,
        "proyecto": {
            "id": str(proyecto.get("ID", "")),
            "nombre": str(proyecto.get("Proyecto", "")),
            "objetivo": str(proyecto.get("Objetivo", "") or ""),
            "requerimientos": str(proyecto.get("Requerimientos", "") or ""),
        },
        "fragmento": {
            "pagina_inicio": int(chunk["page_start"]),
            "pagina_fin": int(chunk["page_end"]),
            "texto": chunk["chunk_text"][:6000],
        }
    }

    # Prompt para forzar JSON válido
    user_instructions = (
        "Devuelve EXCLUSIVAMENTE un JSON válido con estas claves:\n"
        "- relevante (boolean)\n"
        "- justificacion (string, 1-2 frases)\n"
        "- cita_resumen (string, 1 línea)\n"
        "- confianza (number entre 0 y 1)\n\n"
        "Reglas:\n"
        "- Si NO es relevante, pon relevante=false y justificacion/cita_resumen pueden ser vacíos.\n"
        "- No incluyas texto adicional fuera del JSON.\n\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )

    resp = client.responses.create(
        model=LLM_MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_instructions},
        ],
        temperature=0.0
    )

    # Parseo robusto: intenta leer JSON directo; si el modelo mete texto extra, se intenta recortar
    raw = getattr(resp, "output_text", None)
    if raw is None:
        # fallback defensivo (según versión SDK)
        try:
            raw = resp.output[0].content[0].text  # type: ignore
        except Exception:
            raise RuntimeError("No pude obtener output_text del response. Revisa tu versión del SDK de OpenAI.")

    raw = raw.strip()

    try:
        out = json.loads(raw)
    except json.JSONDecodeError:
        # Intenta recortar el primer objeto JSON válido
        m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not m:
            return None
        out = json.loads(m.group(0))

    if not out.get("relevante", False):
        return None

    return {
        "cita_resumen": str(out.get("cita_resumen", "") or "").strip(),
        "justificacion": str(out.get("justificacion", "") or "").strip(),
        "confianza": float(out.get("confianza", 0.0) or 0.0),
    }


# -------------------------
# Representación (query) de cada proyecto para embeddings
# -------------------------
def build_project_queries(df_projects: pd.DataFrame) -> List[str]:
    """
    Para embeddings, representamos cada proyecto con:
    - nombre del proyecto
    - objetivo
    - requerimientos

    No usamos objetivos específicos: si existe la columna y está vacía, no aporta.
    """
    queries = []
    for _, r in df_projects.iterrows():
        q = (
            f"{r['Proyecto']}\n"
            f"OBJETIVO:\n{r.get('Objetivo','')}\n"
            f"REQUERIMIENTOS:\n{r.get('Requerimientos','')}"
        )
        queries.append(q.strip())
    return queries


# -------------------------
# Main
# -------------------------
def main():
    client = get_openai_client()

    # Validaciones de inputs
    if not PROJECTS_XLSX.exists():
        raise FileNotFoundError(f"No existe {PROJECTS_XLSX} (debe estar en la raíz).")
    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(f"No existe la carpeta {DOCUMENTS_DIR}. Crea ./documentos y pon los PDFs.")

    # Lee tabla de proyectos
    df_projects = pd.read_excel(PROJECTS_XLSX)

    # Limpia nombres de columnas por si el Excel trae espacios invisibles
    df_projects.columns = [c.strip() for c in df_projects.columns]

    # Columnas obligatorias mínimas
    required = {"ID", "Proyecto", "Objetivo", "Requerimientos"}
    missing = required - set(df_projects.columns)
    if missing:
        raise ValueError(f"Faltan columnas obligatorias en Proyectos.xlsx: {missing}")

    # Columna opcional (no se usa, pero la toleramos si existe/no existe)
    if "Objetivos_especificos" not in df_projects.columns:
        df_projects["Objetivos_especificos"] = ""

    # Pre-embeddings de proyectos
    project_queries = build_project_queries(df_projects)
    project_embeds = embed_texts(client, project_queries)

    # Lista PDFs
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdfs:
        raise RuntimeError(f"No encontré PDFs en {DOCUMENTS_DIR.resolve()}")

    rows = []
    ref_counter = 1

    for pdf_path in progress(pdfs, total=len(pdfs), desc="Documentos"):
        municipio = infer_municipio_from_filename(pdf_path.name)
        documento = pdf_path.name

        pages = extract_pages_text(pdf_path)
        chunks = chunk_text(pages, CHUNK_CHARS, CHUNK_OVERLAP)
        if not chunks:
            continue

        # Embeddings de chunks (batch)
        chunk_texts = [c["chunk_text"] for c in chunks]
        chunk_embeds = embed_texts(client, chunk_texts)

        # Itera proyectos: para cada proyecto, selecciona top chunks por similitud y valida con LLM
        proj_iter = df_projects.iterrows()
        for p_idx, (_, proj) in enumerate(progress(proj_iter, total=len(df_projects), desc=f"Proyectos ({municipio})")):
            pe = project_embeds[p_idx]

            # Selecciona candidatos por umbral
            sims: List[Tuple[float, int]] = []
            for c_idx, ce in enumerate(chunk_embeds):
                s = cosine(pe, ce)
                if s >= SIM_THRESHOLD:
                    sims.append((s, c_idx))

            if not sims:
                continue

            # Top K por proyecto/documento
            sims.sort(reverse=True, key=lambda x: x[0])
            top = sims[:TOP_K_PER_PROJECT_PER_DOC]

            for sim, c_idx in top:
                c = chunks[c_idx]

                verdict = llm_filter_and_justify(
                    client=client,
                    proyecto=proj.to_dict(),
                    municipio=municipio,
                    documento=documento,
                    chunk=c,
                )
                if verdict is None:
                    continue

                rows.append({
                    "ref_id": ref_counter,
                    "municipio": municipio,
                    "documento": documento,
                    "ruta_documento": str(pdf_path),
                    "tipo_documento": "PLAN_DE_DESARROLLO",
                    "proyecto_id": str(proj["ID"]),
                    "proyecto_nombre": str(proj["Proyecto"]),
                    "pagina_inicio": int(c["page_start"]),
                    "pagina_fin": int(c["page_end"]),
                    "cita_texto": c["chunk_text"],
                    "cita_resumen": verdict["cita_resumen"],
                    "justificacion": verdict["justificacion"],
                    "match_score": float(sim),
                    "metodo_match": "embeddings+llm",
                    "modelo_llm": LLM_MODEL,
                    "modelo_embeddings": EMBED_MODEL,
                    "extractor_version": EXTRACTOR_VERSION,
                    "fecha_extraccion": datetime.now().isoformat(timespec="seconds"),
                })
                ref_counter += 1

                # Pausa pequeña para evitar rate limits
                time.sleep(0.05)

    # Exporta resultados
    df_out = pd.DataFrame(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_excel(OUTPUT_XLSX, index=False)

    print(f"\nListo. Generadas {len(df_out)} referencias -> {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
