#!/usr/bin/env python3
"""
Command-line pipeline to create an academic article from multiple digests.

This script reproduces the notebook workflow in code:
1. Find papers citing a source article with SerpAPI
2. Download PDFs
3. Extract text as Markdown
4. Use an LLM to extract digests (only passages referencing the source)
5. Compose a full article based on all digests

Environment variables:
  SERPAPI_API_KEY   - required for Google Scholar search
  RWB_CORELLM_TOKEN - required for LLM calls
Optional CLI flags allow overriding the defaults.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence, Tuple

import pandas as pd
import pymupdf4llm
import requests
import tqdm
from dotenv import load_dotenv
from openai import OpenAI
from serpapi import GoogleSearch

DEFAULT_SOURCE_QUERY = "Exploring ChatGPT and its impact on society MA Haque, S Li"
DEFAULT_MODEL = "DeepSeek-R1"
DEFAULT_BASE_URL = "https://corellm.wb.ru/deepseek/v1"
DIGEST_SEPARATOR = "\n\n--- DIGEST SEPARATOR ---\n\n"


# ------------------------ Shared utilities --------------------------------- #
def require(value: str | None, name: str) -> str:
    """Fail fast if a required setting is missing."""
    if not value:
        raise ValueError(f"{name} is required. Pass via CLI or environment variable.")
    return value


def chunk_text(text: str, max_chars: int) -> List[str]:
    """
    Split long text into roughly even chunks constrained by max_chars.
    This guards against context-length errors from the LLM API.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV at {path}, but it was not found.")
    return pd.read_csv(path)


def slugify(text: str, max_len: int = 40) -> str:
    """Filesystem-friendly slug for directory names."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text[:max_len] or "query"


# ------------------------ Step 1: citation discovery ----------------------- #
def get_cites_page(cites_id: str, serpapi_key: str, start: int, num: int) -> Tuple[pd.DataFrame, int]:
    params = {
        "engine": "google_scholar",
        "cites": cites_id,
        "api_key": serpapi_key,
        "num": num,
        "start": start,
    }
    search = GoogleSearch(params)
    cite_results = search.get_dict()

    citing_papers = cite_results.get("organic_results", []) or []
    total_results = (cite_results.get("search_information", {}) or {}).get("total_results", 0)

    rows = []
    for paper in citing_papers:
        pdf_url = ""
        for resource in paper.get("resources", []) or []:
            fmt = (resource.get("file_format") or resource.get("type") or "").lower()
            if fmt == "pdf":
                pdf_url = resource.get("link", "")
                break

        pub_info = paper.get("publication_info", {}) or {}
        inline_links = paper.get("inline_links", {}) or {}
        cited_by = inline_links.get("cited_by", {}) or {}

        rows.append(
            {
                "title": paper.get("title"),
                "result_id": paper.get("result_id"),
                "link": paper.get("link"),
                "snippet": paper.get("snippet"),
                "publication_summary": pub_info.get("summary"),
                "cites_id": cited_by.get("cites_id"),
                "pdf_url": pdf_url,
            }
        )

    return pd.DataFrame(rows), int(total_results)


def fetch_citations(
    query: str,
    serpapi_key: str,
    per_page: int = 20,
    max_results: int | None = None,
) -> pd.DataFrame:
    search = GoogleSearch({"engine": "google_scholar", "q": query, "api_key": serpapi_key})
    results = search.get_dict()
    organic_results = results.get("organic_results", []) or []
    if not organic_results:
        raise RuntimeError("No results returned from SerpAPI for the provided query.")

    paper = organic_results[0]
    cites_id = (paper.get("inline_links", {}) or {}).get("cited_by", {}).get("cites_id")
    if not cites_id:
        raise RuntimeError("The top search result does not expose a cites_id for follow-up queries.")

    first_df, total_results = get_cites_page(cites_id, serpapi_key, start=0, num=per_page)
    if max_results is not None:
        total_results = min(total_results, max_results)

    dfs: List[pd.DataFrame] = [first_df]
    fetched = len(first_df)
    start = per_page
    while fetched < total_results:
        remaining = total_results - fetched
        num = per_page if remaining > per_page else remaining
        page_df, _ = get_cites_page(cites_id, serpapi_key, start=start, num=num)
        if page_df.empty:
            break
        dfs.append(page_df)
        fetched += len(page_df)
        start += num

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["title", "result_id", "link", "snippet", "publication_summary", "cites_id", "pdf_url"])


# ------------------------ Step 2: PDF download ----------------------------- #
def download_pdfs(cites_csv: Path, output_dir: Path) -> Tuple[int, int]:
    df = read_csv(cites_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_with_pdfs = df[df["pdf_url"].notna()].copy()
    success, failed = 0, 0
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (compatible; academic-digest/1.0)"}

    for _, row in df_with_pdfs.iterrows():
        pdf_url = row["pdf_url"]
        result_id = row["result_id"]
        filename = f"{result_id}.pdf"
        filepath = output_dir / filename

        if filepath.exists():
            success += 1
            continue

        try:
            with session.get(pdf_url, timeout=30, headers=headers, stream=True) as response:
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            success += 1
            time.sleep(0.5)
        except Exception:
            failed += 1

    return success, failed


# ------------------------ Step 3: PDF extraction --------------------------- #
def extract_pdfs_to_csv(pdf_dir: Path, output_csv: Path) -> pd.DataFrame:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    result_ids: List[str] = []
    pdf_contents: List[str] = []

    for i, pdf_path in enumerate(pdf_files):
        filename = pdf_path.name
        result_id = pdf_path.stem
        print(f"[{i}] Extracting: {filename}")

        try:
            md_text = pymupdf4llm.to_markdown(str(pdf_path))
        except Exception as exc:  # pragma: no cover - defensive
            md_text = f"Error: {exc}"

        result_ids.append(result_id)
        pdf_contents.append(md_text)

    df = pd.DataFrame({"result_id": result_ids, "pdf_content": pdf_contents})
    df["content_length"] = df["pdf_content"].str.len()
    df["has_error"] = df["pdf_content"].str.startswith("Error")
    df.to_csv(output_csv, index=False)
    return df


# ------------------------ Step 4: Digest generation ----------------------- #
def create_llm_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url)


def build_digest_prompt(source_article_name: str, pdf_chunk: str) -> List[dict]:
    system_message = (
        f"You are a helpful assistant. User will give you article text. "
        f"Find passages that reference '{source_article_name}'. "
        "Return only the referenced text. If no references are found, write 'No references found'."
    )
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": pdf_chunk},
    ]


def process_pdf_with_llm(
    pdf_content: str,
    source_article_name: str,
    client: OpenAI,
    model_name: str,
    max_chars: int,
    max_retries: int = 2,
) -> str:
    if not pdf_content or (isinstance(pdf_content, float) and pd.isna(pdf_content)):
        return "No content to process"

    parts = chunk_text(str(pdf_content), max_chars=max_chars)
    part_outputs: List[str] = []

    for part_idx, part in enumerate(parts, start=1):
        last_error: str | None = None
        for attempt in range(max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=build_digest_prompt(source_article_name, part),
                    model=model_name,
                    stream=False,
                    temperature=0.1,
                )
                response = chat_completion.choices[0].message.content or ""
                label = f"[Chunk {part_idx}] " if len(parts) > 1 else ""
                part_outputs.append(label + response.strip())
                break
            except Exception as exc:  # pragma: no cover - network call
                last_error = str(exc)
                time.sleep(1.5)
        else:
            part_outputs.append(f"[Chunk {part_idx}] Error after {max_retries} attempts: {last_error}")

    return "\n\n".join(part_outputs).strip()


def generate_digests(
    extracted_csv: Path,
    source_article_name: str,
    api_key: str,
    base_url: str,
    model_name: str = DEFAULT_MODEL,
    max_workers: int = 6,
    max_chars_per_call: int = 300_000,
) -> Tuple[pd.DataFrame, str]:
    df = read_csv(extracted_csv)
    digest_col_name = f"{model_name}_digest"
    client = create_llm_client(api_key, base_url)

    df[digest_col_name] = None
    non_null_mask = df["pdf_content"].notna() & (df["pdf_content"].astype(str).str.strip() != "")
    rows_to_process = df[non_null_mask]

    args_list = [(idx, row["pdf_content"]) for idx, row in rows_to_process.iterrows()]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_pdf_with_llm,
                pdf_content=pdf_content,
                source_article_name=source_article_name,
                client=client,
                model_name=model_name,
                max_chars=max_chars_per_call,
            ): idx
            for idx, pdf_content in args_list
        }

        with tqdm.tqdm(total=len(futures), desc="Generating digests") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                digest = future.result()
                df.at[idx, digest_col_name] = digest
                pbar.update(1)

    output_csv = extracted_csv.with_name("pdf_extracted_text_with_digests.csv")
    df.to_csv(output_csv, index=False)
    return df, digest_col_name


# ------------------------ Step 5: Article creation ------------------------ #
def collect_valid_digests(df: pd.DataFrame, digest_col: str) -> List[str]:
    if digest_col not in df.columns:
        raise KeyError(f"Column '{digest_col}' not found in digests CSV.")
    mask = (
        df[digest_col].notna()
        & ~df[digest_col].str.contains("No references found", na=False)
        & ~df[digest_col].str.contains("Error after", na=False)
        & (df[digest_col].str.strip() != "")
    )
    return df.loc[mask, digest_col].astype(str).tolist()


def build_article_prompt(source_article_name: str | Sequence[str], digests: Sequence[str]) -> str:
    joined_digests = DIGEST_SEPARATOR.join(digests)
    if isinstance(source_article_name, str):
        source_label = f'the source work "{source_article_name}"'
    else:
        source_label = "the following source works: " + "; ".join(f'"{s}"' for s in source_article_name)

    return textwrap.dedent(
        f"""
        You are an expert academic writer. Using only the digests below, craft a cohesive article
        about {source_label}.

        Requirements:
        - Base the article strictly on the provided digests (each digest references one of the sources).
        - Provide clear sections: Title, Abstract, Background, Cross-paper Evidence, Discussion, Limitations,
          and Conclusion.
        - Highlight how different papers interpret or build upon each source work and where themes overlap.
        - Avoid speculation beyond the digests; if evidence is thin, say so.

        Digests:
        {joined_digests}
        """
    ).strip()


def create_article_from_digests(
    digests: Sequence[str],
    source_article_name: str | Sequence[str],
    api_key: str,
    base_url: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.3,
) -> str:
    if not digests:
        raise RuntimeError("No valid digests available to create an article.")

    client = create_llm_client(api_key, base_url)
    prompt = build_article_prompt(source_article_name, digests)

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are an expert academic researcher who synthesizes multiple sources into coherent articles.",
            },
            {"role": "user", "content": prompt},
        ],
        model=model_name,
        stream=False,
        temperature=temperature,
    )

    return completion.choices[0].message.content or ""


# ------------------------ CLI plumbing ------------------------------------ #
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an academic article from multiple digests.")
    parser.add_argument("--serpapi-api-key", default=os.getenv("SERPAPI_API_KEY"), help="SerpAPI key.")
    parser.add_argument("--llm-api-key", default=os.getenv("RWB_CORELLM_TOKEN"), help="LLM API key.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="LLM model name.")
    parser.add_argument("--llm-base-url", default=DEFAULT_BASE_URL, help="LLM API base URL.")
    parser.add_argument(
        "--query",
        action="append",
        help="Search query for a source article. Repeatable. If omitted, uses the default single query.",
    )
    parser.add_argument("--queries-file", help="Path to file with one query per line.")
    parser.add_argument("--max-serpapi-results", type=int, default=None, help="Limit number of citing papers.")
    parser.add_argument("--max-digest-chars", type=int, default=300_000, help="Max characters per LLM digest call.")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers for digest generation.")
    parser.add_argument("--output-dir", default=".", help="Directory for outputs (CSV, PDFs, article).")
    parser.add_argument("--only-article", action="store_true", help="Skip to article generation using existing digests CSV.")
    parser.add_argument(
        "--digests-csv",
        action="append",
        default=None,
        help="Path to an existing digests CSV. Repeatable for multi-source article-only mode.",
    )
    parser.add_argument("--article-output", default="article.md", help="Where to write the generated article.")
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    serpapi_key = args.serpapi_api_key
    llm_api_key = args.llm_api_key
    model_name = args.model_name
    base_url = args.llm_base_url

    queries: List[str] = []
    if args.query:
        queries.extend(args.query)
    if args.queries_file:
        queries.extend([line.strip() for line in Path(args.queries_file).read_text().splitlines() if line.strip()])
    if not queries:
        queries = [DEFAULT_SOURCE_QUERY]

    source_names: List[str] = list(queries)

    # Article-only mode assumes digests CSV already exists.
    if args.only_article:
        digests_csvs = args.digests_csv or ["pdf_extracted_text_with_digests.csv"]
        all_digests: List[str] = []
        for csv_path in digests_csvs:
            df = read_csv(Path(csv_path))
            digest_col = model_name + "_digest"
            if digest_col not in df.columns:
                fallback = [c for c in df.columns if c.endswith("_digest")]
                digest_col = fallback[0] if fallback else digest_col
            all_digests.extend(collect_valid_digests(df, digest_col=digest_col))

        article = create_article_from_digests(
            all_digests,
            source_names if len(source_names) > 1 else (source_names[0] if source_names else "Combined sources"),
            require(llm_api_key, "RWB_CORELLM_TOKEN"),
            base_url,
            model_name,
        )
        Path(args.article_output).write_text(article, encoding="utf-8")
        print(f"Article written to {args.article_output}")
        return

    try:
        serpapi_key = require(serpapi_key, "SERPAPI_API_KEY")
        llm_api_key = require(llm_api_key, "RWB_CORELLM_TOKEN")
    except ValueError as exc:
        sys.exit(str(exc))

    all_valid_digests: List[str] = []

    for idx, (query, source_name) in enumerate(zip(queries, source_names), start=1):
        run_dir = output_dir / f"run_{idx:02d}_{slugify(query)}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: find citing papers
        print(f"[{idx}/{len(queries)}] Searching for citing papers for '{source_name}'...")
        cites_df = fetch_citations(query, serpapi_key, max_results=args.max_serpapi_results)
        cites_csv = run_dir / "cites.csv"
        cites_df.to_csv(cites_csv, index=False)
        print(f"Found {len(cites_df)} citing papers. Saved to {cites_csv}")

        # Step 2: download PDFs
        pdf_dir = run_dir / "pdfs"
        print("Downloading PDFs...")
        success, failed = download_pdfs(cites_csv, pdf_dir)
        print(f"Downloaded {success} PDFs; failed: {failed}")

        # Step 3: extract PDFs
        print("Extracting PDF contents...")
        extracted_csv = run_dir / "pdf_extracted_text.csv"
        extract_pdfs_to_csv(pdf_dir, extracted_csv)
        print(f"Extraction complete. Saved to {extracted_csv}")

        # Step 4: generate digests
        print("Generating digests with LLM...")
        digests_df, digest_col = generate_digests(
            extracted_csv=extracted_csv,
            source_article_name=source_name,
            api_key=llm_api_key,
            base_url=base_url,
            model_name=model_name,
            max_workers=args.workers,
            max_chars_per_call=args.max_digest_chars,
        )
        digests_csv = extracted_csv.with_name("pdf_extracted_text_with_digests.csv")
        print(f"Digests saved to {digests_csv}")

        valid_digests = [f"[{source_name}] {d}" for d in collect_valid_digests(digests_df, digest_col)]
        all_valid_digests.extend(valid_digests)

    if not all_valid_digests:
        sys.exit("No valid digests found across all queries; cannot create article.")

    article_text = create_article_from_digests(
        digests=all_valid_digests,
        source_article_name=source_names if len(source_names) > 1 else source_names[0],
        api_key=llm_api_key,
        base_url=base_url,
        model_name=model_name,
    )
    article_path = output_dir / args.article_output
    article_path.write_text(article_text, encoding="utf-8")
    print(f"Article written to {article_path}")


if __name__ == "__main__":
    main()

