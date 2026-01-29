# AI Model Recommendations for Legal RAG System

## Executive Summary

This document provides model recommendations for the Hong Kong Property Law RAG system, prioritizing **accuracy** and **legal reasoning capability**. Based on current benchmarks and research, we recommend a tiered approach depending on deployment requirements.

---

## 1. Embedding Models (Retrieval)

### Recommended: Tiered Approach

| Tier | Model | MTEB Score | Legal Suitability | Use Case |
|------|-------|------------|-------------------|----------|
| **Tier 1 (Best)** | `GTE-large` | 63.13 | ⭐⭐⭐⭐⭐ | Production - highest accuracy |
| **Tier 1 (Alt)** | `BGE-large-en-v1.5` | 64.23 | ⭐⭐⭐⭐⭐ | Production - best groundedness |
| **Tier 2** | `BGE-M3` | 62.35 | ⭐⭐⭐⭐ | Multilingual EN/ZH support |
| **Tier 3** | `E5-large-v2` | 62.68 | ⭐⭐⭐⭐ | Good balance speed/accuracy |
| **Specialized** | `Legal-BERT` | N/A | ⭐⭐⭐⭐⭐ | Fine-tune for HK legal terms |

### Why These Models?

Based on [legal RAG benchmarks](https://zilliz.com/ai-faq/what-embedding-models-work-best-for-legal-documents):

1. **GTE-large** - Best overall for legal RAG with balanced semantic relevance and contextual accuracy
2. **BGE-large-en-v1.5** - Achieved perfect scores (1.000) in Response Groundedness and Context Recall
3. **BGE-M3** - Essential for Hong Kong's bilingual legal system (English + Chinese terms)
4. **Legal-BERT** - Domain-specific vocabulary understanding (e.g., "force majeure", "deed of mutual covenant")

### Configuration Recommendation

```python
# Primary: BGE-M3 for bilingual support
EMBEDDING_MODEL = "BAAI/bge-m3"

# Alternative: GTE-large for English-only highest accuracy
# EMBEDDING_MODEL = "thenlper/gte-large"

# For fine-tuning on HK legal corpus
# BASE_MODEL = "nlpaueb/legal-bert-base-uncased"
```

### Fine-Tuning Strategy

For optimal legal retrieval, fine-tune on:
1. **Query-passage pairs** from HK ordinance headings → content
2. **Definition pairs** - legal term → definition text
3. **Cross-reference pairs** - source section → referenced section
4. **Hard negatives** - similar but incorrect sections

---

## 2. Large Language Models (Generation)

### LegalBench Performance (2025-2026)

| Model | LegalBench Score | Strengths | Cost |
|-------|------------------|-----------|------|
| **Gemini 3 Pro** | 87.04% | Best legal benchmark | $$ |
| **GPT-5** | 86.02% | Strong reasoning | $$$ |
| **GPT-5.1** | 85.68% | Balanced | $$$ |
| **Claude Opus 4.5** | ~84%* | Lowest hallucination, safety | $$$ |
| **Claude Sonnet 4.5** | ~82%* | Best cost/performance | $$ |

*Claude models excel in safety and reduced hallucination, critical for legal applications.

### Recommendation by Use Case

| Use Case | Recommended Model | Reasoning |
|----------|-------------------|-----------|
| **Production (Cloud)** | Claude Opus 4.5 or GPT-5 | Highest accuracy, legal safety |
| **Production (Budget)** | Claude Sonnet 4.5 | Best value, low hallucination |
| **Self-Hosted** | LLaMA 3.1 70B | Fine-tunable, no API costs |
| **Edge/Local** | Mistral-7B or Qwen2-7B | Fast inference, reasonable quality |

### Why Claude for Legal Applications?

Per [Anthropic's documentation](https://www.anthropic.com) and [industry analysis](https://www.shakudo.io/blog/top-9-large-language-models):

1. **Constitutional AI** - Designed to be more steerable and less prone to hallucination
2. **Conservative reasoning** - Critical for legal advice where false positives are dangerous
3. **Long context** - 200K tokens supports full ordinance analysis
4. **Compliance-focused** - Recommended for legal/compliance support use cases

### Configuration Recommendation

```python
# Option 1: Claude (Recommended for legal)
LLM_PROVIDER = "anthropic"
LLM_MODEL = "claude-sonnet-4-5-20250514"  # Balance of cost/accuracy
# LLM_MODEL = "claude-opus-4-5-20250514"  # Highest accuracy

# Option 2: OpenAI
LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4o"  # or "gpt-5" when available

# Option 3: Self-hosted
LLM_PROVIDER = "local"
LLM_MODEL = "meta-llama/Llama-3.1-70B-Instruct"
```

---

## 3. Hybrid Retrieval Configuration

### Recommended Weights

```python
RETRIEVAL_CONFIG = {
    "dense_weight": 0.45,      # Semantic similarity (BGE-M3)
    "sparse_weight": 0.35,     # Exact term matching (BM25)
    "kg_weight": 0.20,         # Knowledge graph traversal
}
```

### Legal-Specific BM25 Boosting

```python
LEGAL_TERM_BOOSTS = {
    # Ordinance references (highest priority)
    "cap": 2.5, "cap.": 2.5, "ordinance": 2.0,
    "section": 2.0, "s.": 2.0,

    # Hong Kong specific
    "incorporated owners": 2.5,
    "deed of mutual covenant": 2.5,
    "management committee": 2.0,
    "building management ordinance": 2.5,

    # Legal obligations
    "shall": 1.5, "must": 1.5, "required": 1.5,
    "prohibited": 1.5, "offence": 1.8, "penalty": 1.8,
}
```

---

## 4. Complete Stack Recommendation

### Production Stack (Cloud)

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION STACK                          │
├─────────────────────────────────────────────────────────────┤
│  Embeddings    │  BGE-M3 (bilingual) + GTE-large (accuracy) │
│  Vector DB     │  Qdrant Cloud or Pinecone                  │
│  Sparse Index  │  Elasticsearch                             │
│  Knowledge Graph│ Neo4j AuraDB                              │
│  LLM           │  Claude Opus 4.5 (primary)                 │
│                │  GPT-5 (fallback)                          │
│  Hosting       │  AWS/GCP with GPU instances                │
└─────────────────────────────────────────────────────────────┘
```

### Development/Budget Stack (Self-Hosted)

```
┌─────────────────────────────────────────────────────────────┐
│                    BUDGET STACK                              │
├─────────────────────────────────────────────────────────────┤
│  Embeddings    │  BGE-base-en-v1.5 (smaller, faster)        │
│  Vector DB     │  FAISS (local) or Qdrant (self-hosted)     │
│  Sparse Index  │  rank-bm25 (in-memory)                     │
│  Knowledge Graph│ NetworkX (in-memory)                      │
│  LLM           │  LLaMA 3.1 70B (quantized) or              │
│                │  Claude Sonnet (API, cost-effective)       │
│  Hosting       │  Single GPU server (A100/H100)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Model Performance Expectations

### Retrieval Targets

| Metric | Target | Measurement |
|--------|--------|-------------|
| Recall@10 | >0.92 | Relevant sections in top 10 |
| MRR | >0.75 | Mean Reciprocal Rank |
| Citation Coverage | 100% | All cited sections retrieved |

### Generation Targets

| Metric | Target | Critical For |
|--------|--------|--------------|
| Citation Accuracy | 100% | Legal credibility |
| Factual Accuracy | >95% | Avoiding misinformation |
| Hallucination Rate | <1% | Legal safety |
| Response Groundedness | >0.95 | Verifiability |

---

## 6. Cost Estimates (Monthly)

### Cloud API Costs (1000 queries/day)

| Component | Provider | Est. Cost/Month |
|-----------|----------|-----------------|
| Embeddings | Local/HF | $0 (self-hosted) |
| Vector DB | Qdrant Cloud | $50-100 |
| LLM (Claude Sonnet) | Anthropic | $200-400 |
| LLM (Claude Opus) | Anthropic | $800-1500 |
| LLM (GPT-4o) | OpenAI | $300-600 |

### Self-Hosted Costs

| Component | Hardware | Est. Cost/Month |
|-----------|----------|-----------------|
| GPU Server (A100 80GB) | Cloud | $2000-3000 |
| GPU Server (RTX 4090) | On-prem | $50 (electricity) |
| Storage (500GB SSD) | Cloud | $50-100 |

---

## 7. Implementation Priority

### Phase 1: MVP (Weeks 1-2)
- [x] BGE-base-en-v1.5 embeddings
- [x] FAISS vector store
- [x] BM25 sparse retrieval
- [ ] Claude Sonnet 4.5 for generation

### Phase 2: Enhancement (Weeks 3-4)
- [ ] Upgrade to BGE-M3 for bilingual
- [ ] Fine-tune on HK legal corpus
- [ ] Add GTE-large as secondary retriever
- [ ] Implement reranking with cross-encoder

### Phase 3: Production (Weeks 5-6)
- [ ] Migrate to Qdrant Cloud
- [ ] Add Claude Opus for complex queries
- [ ] Implement fallback chain (Opus → Sonnet → GPT-4o)
- [ ] Deploy evaluation pipeline

---

## 8. Key References

- [Best Embedding Models for Legal Documents](https://zilliz.com/ai-faq/what-embedding-models-work-best-for-legal-documents) - Zilliz
- [LegalBench: Legal Reasoning Benchmark](https://hazyresearch.stanford.edu/legalbench/) - Stanford HAI
- [Top LLMs 2026](https://www.shakudo.io/blog/top-9-large-language-models) - Shakudo
- [Open Source Embedding Models Benchmark](https://research.aimultiple.com/open-source-embedding-models/) - AIMultiple
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Hugging Face

---

*Document Version: 1.0*
*Last Updated: 2026-01-29*
