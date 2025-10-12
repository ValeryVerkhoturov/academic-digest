# Academic Digest Analysis System

A pipeline for analyzing academic papers and generating summaries based on how they reference a specific source article. This system uses Google Scholar search, PDF processing, and Large Language Models (LLMs) to create detailed academic insights.

## Prerequisites

- Python 3.12+
- Poetry (for dependency management)
- API Keys:
  - `SERPAPI_API_KEY` - For Google Scholar search
  - `RWB_CORELLM_TOKEN` - For LLM processing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ValeryVerkhoturov/academic-digest
cd academic-digest
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set up environment variables:
```bash
echo "SERPAPI_API_KEY=your_serpapi_key_here" > .env
echo "RWB_CORELLM_TOKEN=your_corellm_token_here" >> .env
```

## Notebook Workflow

The project consists of 5 sequential notebooks that form a complete analysis pipeline:

### 1. `1_get_cite_urls.ipynb` - Citation Discovery
**Purpose**: Search Google Scholar for papers that cite the source article

**What it does**:
- Uses SerpAPI to search Google Scholar
- Searches for papers referencing to user query (ex. 'Exploring ChatGPT and its impact on society MA Haque, S Li')
- Extracts paper metadata and saves results to `cites.csv`

**Output**: `cites.csv` - Contains paper metadata and PDF URLs

### 2. `2_fetch_pdfs.ipynb` - PDF Download
**Purpose**: Download PDF files of the discovered papers

**What it does**:
- Reads `cites.csv` to get PDF URLs
- Downloads PDF files to `pdfs/` folder

**Output**: `pdfs/` folder - Contains downloaded PDF files

### 3. `3_extract_pdfs.ipynb` - Text Extraction
**Purpose**: Extract text content from PDF files

**What it does**:
- Processes all PDF files in `pdfs/` folder
- Extracts text as Markdown

**Output**: `pdf_extracted_text.csv` - Contains extracted text content

### 4. `4_llm_process_of_extraction.ipynb` - Reference Analysis
**Purpose**: Use LLMs to find references to the source article in extracted text

**What it does**:
- Loads extracted text from CSV
- Generates digests for each paper with DeepSeek-R1

**Output**: `pdf_extracted_text_with_digests.csv` - Contains original text + LLM digests

### 5. `5_digest_summary.ipynb` - Summary Generation
**Purpose**: Generate summary of the source article based on references

**What it does**:
- Analyzes all valid digests from previous step
- Generates summary of the source article with DeepSeek-R1