### 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document processing and knowledge retrieval. It extracts text and tables from PDFs using the **Unstructured** library, stores raw document chunks in **Redis**, and indexes extracted embeddings in **PGVector** for semantic search. The system leverages **MultiVector Retriever** for context retrieval before querying **Gemini Flash**, and includes a biomedical ontology mapping pipeline that links extracted entities to **UMLS / SNOMED-CT**.

![My Image](https://github.com/Mercytopsy/pdf-rag-chatbot-streamlit/blob/main/Architectural%20Diagram.png)


### 🚀 Features

- **Unstructured Document Processing**: Extracts text and tables from PDFs.
- **Redis for Raw Storage**: Stores and retrieves raw document chunks efficiently for persistent storage.
- **PGVector for Vector Storage**: Indexes and retrieves high-dimensional embeddings for similarity search.
- **MultiVector Retriever**: Optimized for retrieving contextual information from multiple sources.
- **Gemini Integration**: Uses **gemini-3-flash-preview** for summarization and RAG responses, and **gemini-embedding-001** for embeddings.
- **Parallel Summarization**: Up to 8 concurrent LLM requests via `ThreadPoolExecutor` with live chunk counter.
- **Source Metadata Tracking**: Stores `filename`, `page_number`, `chunk_type` in `langchain_pg_embedding.cmetadata`.
- **Incremental Upload Progress**: Live 6-step progress bar in the sidebar during PDF processing.
- **PDF Library**: Sidebar radio list of all indexed PDFs with Redis key display; updates automatically after indexing.
- **Document Info Tab**: Auto-generates Summary, Table of Contents, and Key Entities (Anatomical / Materials / Clinical / Technical) per PDF.
- **UMLS / SNOMED-CT Entity Mapping**: Maps extracted entities using semantic similarity + UMLS API; suggests SNOMED-CT parent nodes for unmapped terms.
- **Entity Relationship Graph**: Extracts pairwise relationships via RAG and renders an interactive force-directed graph (pyvis).
- **Clear All Data**: One-click button to wipe Redis and PGVector for a fresh start.

### 🛠️ Tech Stack

#### Programming Language
- Python

#### Libraries
- `unstructured`
- `langchain-postgres` (PGVector)
- `langchain-google-genai`
- `redis`
- `langchain`
- `streamlit`
- `pyvis` (interactive graph rendering)
- `requests` + `numpy` (UMLS API + cosine similarity)

#### Databases
- **Redis**: Raw document chunk storage (keyed by `doc_id`) + PDF index (keyed by `pdf:<sha256>`)
- **PostgreSQL + PGVector**: Embedding vectors + LLM summaries + source metadata

#### LLM & Embeddings
- **LLM**: `gemini-3-flash-preview` (via Google Generative AI API)
- **Embeddings**: `gemini-embedding-001`

### ⚙️ Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   pip install pyvis
   ```

2. Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PG_USER=postgres
   PG_PASSWORD=your_password
   PG_HOST=localhost
   PG_PORT=5432
   PG_DATABASE=postgres
   UMLS_API_KEY=your_umls_api_key
   ```

3. Start Redis and PostgreSQL, then run:
   ```bash
   streamlit run RAG_with_streamlit.py
   ```

---

### 🧩 Code 설명

#### 전체 흐름

```
PDF 업로드
    ↓
파싱 (Unstructured) → 텍스트/테이블 청크 추출 + 소스 메타데이터 수집
    ↓
병렬 요약 (Gemini, 최대 8개 동시) — progress bar 실시간 업데이트
    ↓
임베딩 생성 → PGVector 저장 (요약문 + metadata)
원본 청크 → Redis 저장 (UUID 키)
    ↓
사용자 질문 → 벡터 검색 → 원본 청크 조회 → LLM 응답

[Document Info 탭]
선택된 PDF → Summary / TOC / Entities (RAG, filename 필터)
    ↓
UMLS Mapping: Entities → UMLS API 검색 → 임베딩 코사인 유사도 → Semantic Type 필터
    ↓ (미매핑 엔티티)
LLM → SNOMED-CT 부모 노드 추천
    ↓
Relationship 추출 (RAG) → pyvis 인터랙티브 그래프
```

---

#### 주요 함수 설명 — `RAG_with_streamlit.py`

**`load_pdf_data(file_path)`**
- `unstructured` 라이브러리로 PDF를 파싱
- `strategy="fast"` : Tesseract 없이 빠르게 텍스트/테이블 추출
- `chunking_strategy="by_title"` : 제목 단위로 청크 분리 (최대 10,000자)

**`get_pdf_hash(pdf_path)`**
- PDF 파일의 SHA-256 해시 생성
- Redis에 저장하여 동일 파일 재업로드 시 처리 스킵

**`summarize_text_and_tables(text, tables, status_text, progress_bar)`**
- 각 청크를 `gemini-3-flash-preview`로 요약
- `ThreadPoolExecutor(max_workers=8)`로 최대 8개 병렬 요청
- `as_completed()` 루프(메인 스레드)에서 UI 업데이트 — 백그라운드 스레드에서 Streamlit 위젯 접근 방지
- Rate limit 대응: 자동 재시도 (최대 5회, exponential backoff)
- progress bar가 step 3→4 사이를 청크 단위로 sub-progress 업데이트

**`initialize_retriever(filename_filter=None)`**
- `PGVector` : 요약문의 임베딩 벡터 저장소 (PostgreSQL)
- `RedisStore` : 원본 청크 저장소 (Redis)
- `MultiVectorRetriever` : 두 저장소를 `doc_id`로 연결
- `filename_filter` 지정 시 `search_kwargs={"filter": {"filename": ...}}`를 주입해 특정 PDF에서만 검색

**`store_docs_in_retriever(..., text_meta, table_meta)`**
- 요약문 + 임베딩 → `PGVector` (`langchain_pg_embedding` 테이블)
- 원본 텍스트/테이블 → `Redis` (UUID 키)
- 각 Document에 `filename`, `page_number`, `chunk_type` 메타데이터 포함

**`chat_with_llm(retriever)`**
- RAG chain 구성:
  ```
  질문 → retriever (벡터 검색) → 원본 청크 조회 → 프롬프트 구성 → LLM → 응답
  ```

**`get_document_info(filename)`**
- filename 필터가 걸린 retriever로 RAG chain을 구성해 3가지 쿼리 실행
  - Summary: 3~5문장 요약
  - TOC: 번호가 매겨진 목차
  - Entities: Anatomical / Materials & Substances / Clinical / Technical 카테고리별 bullet list

**`extract_relationships(entities, filename)`**
- 동일 RAG chain으로 엔티티 목록을 전달해 관계를 JSON 배열로 추출
- `re.search(r"\[[\s\S]*\]")`로 LLM 응답에서 JSON 부분만 파싱
- 반환 형식: `[{"source": "...", "relation": "...", "target": "..."}]`

**`render_entity_graph(umls_results, relationships)`**
- pyvis `Network`로 force-directed 그래프 생성
- 노드 색상: 🟢 `#00b894` (mapped) / 🟡 `#fdcb6e` (partial) / 🔴 `#e17055` (new)
- 엣지에 관계 레이블 표시, hover tooltip으로 CUI / Semantic Type / SNOMED 코드 확인
- 임시 HTML 파일 생성 후 `streamlit.components.v1.html()`로 삽입

**`process_pdf(file_upload, progress_bar, status_text)`**
- PDF 전체 처리 파이프라인 (6단계 progress bar)
  1. 📄 파일 저장
  2. 🔌 벡터 스토어 연결
  3. 🔍 PDF 파싱 + 메타데이터 수집
  4. 📝 청크 병렬 요약 (sub-progress)
  5. 💾 임베딩 인덱싱
  6. ✅ 완료 → `st.rerun()`으로 PDF 라이브러리 갱신

**`main()`**
- Streamlit UI 구성
  - **사이드바**: PDF 라이브러리 (radio 선택 + Redis key 표시) / Clear All Data 버튼 / PDF 업로드 위젯
  - **📄 Document Info 탭**: Summary / TOC / Entities → UMLS 매핑 테이블 → 관계 그래프
  - **💬 Chat 탭**: 채팅 입력, 히스토리 표시, 응답 생성

---

#### 주요 함수 설명 — `umls_client.py`

**UMLS 인증 (CAS 2단계)**
1. `get_tgt()`: API Key로 TGT URL 획득 (8시간 유효)
2. `_service_ticket(tgt_url)`: TGT로 일회용 ST 발급 — 매 API 호출마다 새 ST 필요

**`search_umls(term, tgt_url)`**
- UMLS `search/current` 엔드포인트로 후보 개념 최대 10개 검색
- `ui == "NONE"` 결과 필터링

**`get_semantic_types(cui, tgt_url)`**
- CUI에 해당하는 Semantic Type 약어 목록 반환 (예: T047, T074)
- 필터 기준 15종: Body Part, Tissue, Disease, Procedure, Medical Device, Biomedical/Dental Material 등

**`get_snomed_atoms(cui, tgt_url)`**
- `sabs=SNOMEDCT_US` 필터로 SNOMED-CT 코드와 선호명 반환

**`map_entity(term, tgt_url, embeddings, threshold=0.72)`**
핵심 매핑 로직:
1. UMLS에서 후보 10개 검색
2. `gemini-embedding-001`로 쿼리 텀과 후보명 임베딩 생성
3. 코사인 유사도 계산 → 가장 높은 후보 선택
4. 해당 CUI의 Semantic Type이 `RELEVANT_STYS`에 속하는 경우만 통과
5. `similarity ≥ 0.72` → **mapped**, 미만 → **partial**, 후보 없음 → **new**

**`suggest_snomed_parent(term, llm)`**
- UMLS에 없는 새 엔티티에 대해 LLM에게 SNOMED-CT 부모 노드 추천 요청
- 반환: `{"parent_name": "...", "parent_id": "...", "rationale": "..."}`

**`map_entities_to_umls(terms, progress_cb)`**
- 전체 파이프라인: TGT 획득 → 엔티티별 순차 매핑 → 미매핑 시 부모 추천
- `progress_cb(i, total, term)`: UI progress bar 업데이트 콜백

**`parse_entity_terms(entities_text)`**
- LLM이 생성한 그룹별 bullet list에서 순수 용어만 추출
- 괄호/콜론/대시 이하 설명 제거 + 중복 제거 (순서 보존)

---

#### 데이터 저장 구조

| 저장소 | 키 형식 | 내용 |
|--------|---------|------|
| Redis | `pdf:<sha256>` | 처리 완료 표시 + 파일명 |
| Redis | `<uuid>` | 원본 텍스트/테이블 청크 |
| PostgreSQL | `langchain_pg_collection` | 컬렉션 메타데이터 |
| PostgreSQL | `langchain_pg_embedding` | 요약문 + 임베딩 벡터 + `doc_id` + `filename` + `page_number` + `chunk_type` |

---

### 🔄 Modifications from Original

| Area | Original | Updated |
|------|----------|---------|
| LLM | `ChatOpenAI` (`gpt-4o-mini`) | `ChatGoogleGenerativeAI` (`gemini-3-flash-preview`) |
| Embeddings | `OpenAIEmbeddings` | `GoogleGenerativeAIEmbeddings` (`gemini-embedding-001`) |
| PDF strategy | `hi_res` (requires Tesseract) | `fast` (no system OCR dependency) |
| Database | `localhost:6024` (langchain user) | `localhost:5432` (configurable via `.env`) |
| Poppler | Not bundled | Bundled in `poppler/` folder, added to PATH at runtime |
| Summarization | Sequential + 1s sleep | Parallel (8 workers) via `ThreadPoolExecutor` |
| Source metadata | Not tracked | `filename`, `page_number`, `chunk_type` in `cmetadata` |
| Upload UX | Silent processing on first chat | Incremental 6-step progress bar on upload |
| PDF history | Not shown | Radio list in sidebar with Redis key; auto-refreshes after indexing |
| Document Info | Not available | Summary / TOC / Entities per PDF via filtered RAG |
| UMLS mapping | Not available | Semantic similarity + UMLS API + SNOMED-CT lookup |
| New entity handling | Not available | LLM suggests SNOMED-CT parent node |
| Relationship graph | Not available | RAG extraction + pyvis interactive graph |
| Data management | No reset | Clear All Data button wipes Redis + PGVector |
| Dependencies removed | `nvidia-nccl-cu12`, `triton` (Linux-only) | Removed from `requirements.txt` |
