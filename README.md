# persian-cultural-rag-agent


a basic for making a RAG system step by step 


## Problem Statement

برای اینکه گردشگران بتوانند به اطلاعات دقیق تری از بناهای تاریخی ایران 
دستری داشته باشند و با ان اشنا شوند


## architecture
```mermaid
graph TD
    subgraph Ingestion Pipeline [بخش آماده‌سازی داده]
        A[Raw Data] --> B[Cleaning & Schema Validation]
        B --> C[Chunking]
        C --> D[Embedding Model]
        D --> E[(Vector DB / Qdrant)]
    end

    subgraph Retrieval Pipeline [بخش جستجو]
        F[User Query] --> G[Embed Query]
        G --> H[Vector Search]
        E --> H
        H --> I[Top-K Context Chunks]
    end



## schema

{
  "config": {
    "tokenizer": "cl100k_base",
    "child_target_tokens": 400,
    "child_max_tokens": 500,
    "child_overlap_tokens": 40,
    "parent_target_tokens": 900,
    "parent_max_tokens": 1200
  },

  "stats": {
    "pages": 4130,
    "semantic_units": 13809,
    "parents": 16043,
    "children": 25268,
    "max_parent_tokens": 1200,
    "max_child_tokens": 500,
    "mean_parent_tokens": 477.3,
    "mean_child_tokens": 303.6
  },

  "parents": [
    {
      "parent_id": "10008#sec_0#parent_000",
      "semantic_unit_id": "10008#sec_0",
      "page_id": "10008",
      "page_title": "عنوان صفحه",
      "page_url": "https://...",
      "unit_type": "section",
      "section_index": 0,
      "section_heading": "عنوان بخش",
      "section_level": "h2",
      "parent_index": 0,
      "text": "متن parent",
      "tokens": 850
    }
  ],

  "children": [
    {
      "chunk_id": "10008#sec_0#parent_000#child_000",
      "parent_id": "10008#sec_0#parent_000",
      "semantic_unit_id": "10008#sec_0",
      "page_id": "10008",
      "page_title": "عنوان صفحه",
      "page_url": "https://...",
      "unit_type": "section",
      "section_index": 0,
      "section_heading": "عنوان بخش",
      "section_level": "h2",
      "parent_index": 0,
      "child_index": 0,
      "text": "متن child",
      "tokens": 380
    }
  ]
}