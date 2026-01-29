# Legal RAG System Architecture Plan
## Property Owner & Incorporated Owners Rights Advisory System

---

## 1. Executive Summary

This document outlines the architecture for a Retrieval-Augmented Generation (RAG) system specifically designed for Hong Kong property law, focusing on the rights and obligations of Incorporated Owners (IO) and individual property owners. **Accuracy and comprehensiveness are the primary objectives**.

### Dataset Scope
- **47 Ordinances** covering Building Management, Fire Safety, Urban Renewal
- **~60 MB** of structured XML legal data
- **49 XML files** with rich semantic markup

---

## 2. Key Challenges for Legal RAG Accuracy

### 2.1 Legal-Specific Challenges

| Challenge | Impact | Mitigation Strategy |
|-----------|--------|---------------------|
| **Precise Citation Required** | Legal advice must reference exact sections | Preserve section IDs and hierarchical structure |
| **Cross-References** | Laws reference other laws frequently | Build knowledge graph of references |
| **Temporal Validity** | Laws change over time | Track `startPeriod`, `status` attributes |
| **Definitions are Critical** | Legal terms have precise meanings | Create dedicated definition index |
| **Hierarchical Structure** | Section > Subsection > Paragraph | Maintain parent-child relationships in chunks |
| **Bilingual Terms** | English/Chinese terminology | Index both language variants |

### 2.2 Why Standard RAG Fails for Legal

1. **Chunking destroys context** - A subsection without its parent section loses meaning
2. **Semantic search misses exact terms** - "Building Management Ordinance" vs "BMO" vs "Cap. 344"
3. **No citation tracking** - Cannot verify which specific section supports an answer
4. **Cross-reference blindness** - References like "see section 3A" become meaningless

---

## 3. Proposed Architecture: Hierarchical Legal RAG

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER QUERY                                    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    QUERY UNDERSTANDING LAYER                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐  │
│  │ Legal Term  │  │ Query Type  │  │ Referenced                  │  │
│  │ Extraction  │  │ Classifier  │  │ Ordinance Detection         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID RETRIEVAL LAYER                            │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐   │
│  │ Dense Retrieval  │  │ Sparse Retrieval │  │ Knowledge Graph  │   │
│  │ (Semantic)       │  │ (BM25/Exact)     │  │ Traversal        │   │
│  │                  │  │                  │  │                  │   │
│  │ - Section chunks │  │ - Legal terms    │  │ - Cross-refs     │   │
│  │ - Definitions    │  │ - Cap numbers    │  │ - Definitions    │   │
│  │ - Context        │  │ - Section IDs    │  │ - Related laws   │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTEXT ASSEMBLY LAYER                            │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ 1. Hierarchical Context Expansion                               ││
│  │    - Add parent sections for retrieved subsections              ││
│  │    - Include relevant definitions                               ││
│  │                                                                 ││
│  │ 2. Cross-Reference Resolution                                   ││
│  │    - Fetch referenced sections from other ordinances            ││
│  │    - Expand "see section X" references                          ││
│  │                                                                 ││
│  │ 3. Context Ranking & Deduplication                              ││
│  │    - Rank by relevance + legal authority                        ││
│  │    - Remove duplicate content                                   ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    GENERATION LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Legal-Specialized LLM with:                                     ││
│  │ - Instruction to cite specific sections                         ││
│  │ - Chain-of-thought for legal reasoning                          ││
│  │ - Uncertainty quantification                                    ││
│  │ - Structured output with citations                              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    VERIFICATION LAYER                                │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ - Citation verification (do cited sections exist?)              ││
│  │ - Hallucination detection                                       ││
│  │ - Confidence scoring                                            ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Preprocessing Pipeline

### 4.1 XML Parsing Strategy

```python
# Key elements to extract from HKEL XML
EXTRACTION_SCHEMA = {
    "document_metadata": {
        "source": "meta",
        "fields": ["docName", "docType", "docNumber", "docStatus", "dc:date"]
    },
    "content_units": {
        "part": {"id", "name", "heading", "num"},
        "section": {"id", "name", "heading", "num", "startPeriod", "status"},
        "subsection": {"id", "name", "num", "content"},
        "paragraph": {"id", "name", "num", "content"},
        "definition": {"name", "term", "content", "term_zh"}
    },
    "cross_references": {
        "ref": {"href", "text"}
    }
}
```

### 4.2 Preprocessing Steps

1. **Parse XML** → Extract structured content preserving hierarchy
2. **Build Section Index** → Map section IDs to full paths (Cap.344 > Part I > s.2 > (1))
3. **Extract Definitions** → Create dedicated definition lookup table
4. **Resolve Cross-References** → Build reference graph
5. **Generate Clean Text** → Strip XML but preserve structure markers
6. **Create Chunks** → Section-based chunking with metadata

---

## 5. Chunking Strategy for Legal Accuracy

### 5.1 Hierarchical Chunking (Recommended)

**Principle**: Never break legal semantic units

```
Level 1: ORDINANCE (Cap. 344 - Building Management Ordinance)
    │
    ├── Level 2: PART (Part I - Short Title and Interpretation)
    │       │
    │       ├── Level 3: SECTION (s.2 - Interpretation) ← PRIMARY CHUNK UNIT
    │       │       │
    │       │       ├── Level 4: SUBSECTION ((1), (2), etc.)
    │       │       │       │
    │       │       │       └── Level 5: PARAGRAPH ((a), (b), etc.)
    │       │       │
    │       │       └── DEFINITIONS (embedded within sections)
    │       │
    │       └── Level 3: SECTION (s.3 - Meeting of owners)
    │
    └── Level 2: PART (Part II - Incorporation of Owners)
```

### 5.2 Chunk Types

| Chunk Type | Content | Use Case |
|------------|---------|----------|
| **Section Chunk** | Full section with all subsections | Primary retrieval unit |
| **Definition Chunk** | Single term definition | Term lookup |
| **Cross-Reference Chunk** | Section + all referenced sections | Context expansion |
| **Summary Chunk** | Part/Ordinance overview | High-level queries |

### 5.3 Chunk Metadata Schema

```json
{
    "chunk_id": "cap344_s2_1",
    "ordinance": "Cap. 344",
    "ordinance_name": "Building Management Ordinance",
    "part": "Part I",
    "part_name": "Short Title and Interpretation",
    "section": "s.2",
    "section_name": "Interpretation",
    "subsection": "(1)",
    "hierarchy_path": "Cap.344 > Part I > s.2 > (1)",
    "effective_date": "2025-07-13",
    "status": "operational",
    "parent_chunk_id": "cap344_s2",
    "child_chunk_ids": ["cap344_s2_1_a", "cap344_s2_1_b"],
    "cross_references": ["cap588_s2_1", "cap128"],
    "definitions_used": ["owner", "building", "corporation"],
    "text": "...",
    "text_length": 1500,
    "embedding": [...]
}
```

---

## 6. Embedding Strategy

### 6.1 Model Selection for Legal Domain

| Model | Pros | Cons | Recommendation |
|-------|------|------|----------------|
| **BGE-M3** | Multilingual, dense+sparse | Large | Best for EN/ZH bilingual |
| **E5-Large-v2** | Strong legal performance | English-only | Good for English queries |
| **Legal-BERT** | Domain-specific | Older architecture | Fine-tune base |
| **OpenAI ada-002** | Easy to use | API dependency | Baseline comparison |

### 6.2 Recommended: Fine-tuned BGE-M3

**Why BGE-M3:**
- Native support for hybrid (dense + sparse) retrieval
- Multilingual (English + Chinese terms in HK law)
- State-of-the-art performance on legal benchmarks

### 6.3 Fine-tuning Strategy

**Training Data Generation:**
1. Generate query-passage pairs from:
   - Section headings → Section content
   - Definitions → Term usage in sections
   - Cross-references → Source and target sections

2. Hard negatives:
   - Similar sections from different ordinances
   - Repealed vs. current versions
   - Different subsections of same section

---

## 7. Retrieval Strategy

### 7.1 Hybrid Retrieval Pipeline

```python
def retrieve(query: str, top_k: int = 10) -> List[Chunk]:
    # Stage 1: Dense retrieval (semantic similarity)
    dense_results = dense_index.search(
        embed(query),
        top_k=top_k * 2
    )

    # Stage 2: Sparse retrieval (exact term matching)
    sparse_results = bm25_index.search(
        query,
        top_k=top_k * 2,
        boost_fields=["section_name", "definitions"]
    )

    # Stage 3: Knowledge graph expansion
    kg_results = knowledge_graph.traverse(
        extract_legal_entities(query),
        max_hops=2
    )

    # Stage 4: Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(
        dense_results,
        sparse_results,
        kg_results,
        weights=[0.4, 0.3, 0.3]
    )

    # Stage 5: Hierarchical context expansion
    expanded_results = expand_context(fused_results)

    return expanded_results[:top_k]
```

### 7.2 Context Expansion Rules

1. **Parent Inclusion**: If subsection retrieved, include parent section
2. **Definition Injection**: Include definitions for legal terms in retrieved chunks
3. **Cross-Reference Fetch**: Fetch referenced sections up to 1 hop
4. **Related Sections**: Include adjacent sections if same topic

---

## 8. Knowledge Graph Design

### 8.1 Node Types

```
ORDINANCE ─────────────── Contains ────────────── PART
    │                                               │
    │                                          Contains
    │                                               │
    └──── References ────── SECTION ◄──────────────┘
                               │
                          Contains
                               │
                          DEFINITION
```

### 8.2 Edge Types

| Edge Type | From | To | Example |
|-----------|------|-----|---------|
| `CONTAINS` | Ordinance | Part | Cap.344 → Part I |
| `CONTAINS` | Part | Section | Part I → s.2 |
| `REFERENCES` | Section | Section | s.2 → s.3A |
| `DEFINES` | Section | Definition | s.2 → "owner" |
| `AMENDS` | Ordinance | Section | 27 of 1993 → s.2 |
| `RELATED_TO` | Section | Section | (semantic similarity) |

### 8.3 Graph Database

**Recommended: Neo4j or NetworkX**

```cypher
// Example Cypher query for cross-reference traversal
MATCH (s:Section {id: 'cap344_s3'})-[:REFERENCES*1..2]->(related:Section)
RETURN s, related
```

---

## 9. Generation Layer

### 9.1 Model Options

| Option | Description | Accuracy Potential |
|--------|-------------|-------------------|
| **Fine-tuned LLaMA 3** | Domain-adapted open model | High |
| **GPT-4 with legal prompts** | Strong reasoning | Very High |
| **Claude 3 Opus** | Best instruction following | Very High |
| **Mistral-7B Legal** | Efficient, fine-tunable | Medium-High |

### 9.2 Legal-Specific Prompting

```markdown
## System Prompt for Legal RAG

You are a Hong Kong property law assistant specializing in:
- Building Management Ordinance (Cap. 344)
- Rights of Incorporated Owners
- Fire safety compliance
- Urban renewal procedures

CRITICAL INSTRUCTIONS:
1. ALWAYS cite specific sections (e.g., "Under s.14(2) of Cap. 344...")
2. If information is not in the provided context, say "I cannot find this in the provided legal texts"
3. Distinguish between MANDATORY requirements and RECOMMENDED practices
4. Note if any cited law has been amended or repealed
5. For complex matters, recommend seeking professional legal advice

FORMAT YOUR RESPONSE:
- Start with a direct answer
- Provide legal basis with citations
- Note any exceptions or conditions
- List related provisions if relevant
```

### 9.3 Output Schema

```json
{
    "answer": "string",
    "citations": [
        {
            "ordinance": "Cap. 344",
            "section": "s.14(2)",
            "text_excerpt": "...",
            "relevance": "This section establishes..."
        }
    ],
    "confidence": 0.85,
    "caveats": ["This applies only to buildings with IO formed after 1993"],
    "related_sections": ["s.14(1)", "s.15"],
    "recommend_professional_advice": false
}
```

---

## 10. Evaluation Metrics

### 10.1 Retrieval Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Recall@10** | >0.90 | % of relevant sections in top 10 |
| **MRR** | >0.70 | Mean Reciprocal Rank |
| **Citation Coverage** | 100% | All cited sections must be retrieved |

### 10.2 Generation Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| **Citation Accuracy** | 100% | Cited sections must exist and be relevant |
| **Factual Accuracy** | >95% | Legal statements must match source |
| **Hallucination Rate** | <2% | Fabricated citations or provisions |
| **Answer Completeness** | >90% | All relevant provisions mentioned |

### 10.3 Evaluation Dataset

Create a gold-standard Q&A dataset with:
1. **50+ questions** across all ordinance categories
2. **Expert-annotated answers** with required citations
3. **Edge cases**: Repealed sections, cross-ordinance queries
4. **Negative examples**: Questions not answerable from dataset

---

## 11. Implementation Roadmap

### Phase 1: Data Pipeline (Week 1-2)
- [ ] XML parser for HKEL format
- [ ] Hierarchical chunking implementation
- [ ] Definition extraction
- [ ] Cross-reference graph builder
- [ ] Data validation and quality checks

### Phase 2: Indexing (Week 2-3)
- [ ] Set up vector database (Qdrant/Weaviate)
- [ ] Generate embeddings for all chunks
- [ ] Build BM25 index
- [ ] Create knowledge graph in Neo4j
- [ ] Implement hybrid retrieval

### Phase 3: Retrieval Optimization (Week 3-4)
- [ ] Fine-tune embedding model
- [ ] Tune retrieval weights
- [ ] Implement context expansion
- [ ] Build evaluation benchmark
- [ ] Iterate on retrieval quality

### Phase 4: Generation (Week 4-5)
- [ ] Develop legal prompts
- [ ] Implement citation verification
- [ ] Build output formatter
- [ ] Add confidence scoring
- [ ] Integration testing

### Phase 5: Evaluation & Refinement (Week 5-6)
- [ ] Run full evaluation suite
- [ ] Error analysis
- [ ] Fine-tune based on failures
- [ ] User acceptance testing
- [ ] Documentation

---

## 12. Technology Stack

### Recommended Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| **Vector DB** | Qdrant | Native hybrid search, metadata filtering |
| **Embedding** | BGE-M3 (fine-tuned) | Multilingual, hybrid |
| **Graph DB** | Neo4j | Mature, good for legal relationships |
| **BM25** | Elasticsearch or Built-in | Legal term exact matching |
| **LLM** | LLaMA 3 70B or GPT-4 | Best reasoning for legal |
| **Framework** | LlamaIndex or LangChain | RAG orchestration |
| **Evaluation** | RAGAS + Custom | Legal-specific metrics |

### PyTorch Integration

Since this is `legal_advisor_pytorch`, key PyTorch components:
- Custom embedding fine-tuning with PyTorch
- Legal NER model for entity extraction
- Query classification model
- Confidence calibration model

---

## 13. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Hallucinated citations** | Post-generation citation verification |
| **Outdated information** | Track effective dates, flag amendments |
| **Missing cross-references** | Knowledge graph ensures completeness |
| **Incorrect legal interpretation** | Disclaimer + recommend professional advice |
| **Query out of scope** | Query classifier to detect and refuse |

---

## 14. Next Steps

1. **Immediate**: Implement XML parser and data preprocessing
2. **Short-term**: Build chunking pipeline and generate embeddings
3. **Medium-term**: Develop hybrid retrieval and knowledge graph
4. **Long-term**: Fine-tune models and build evaluation framework

---

*Document Version: 1.0*
*Created: 2026-01-29*
*Dataset: HKEL Property Owner Ordinances (47 files, 60MB)*
