# Development Research Papers

This directory contains the development corpus used to test end-to-end ingestion,
chunking, embedding, knowledge extraction, and provenance preservation.

> **Note on experimental scope**: These papers serve as the initial development
> and verification corpus. Formal benchmark evaluation will be conducted on
> GraphRAG-Bench as a separate, controlled dataset.

## Target Papers

1. **Lewis et al. (2020)**
   - **Title**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
   - **Authors**: Patrick Lewis, Ethan Perez, Aleksandara Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, Douwe Kiela
   - **Venue**: NeurIPS 2020 / arXiv:2005.11401
   - **Expected filename**: `data/papers/lewis_2020_rag.pdf`
   - **Open Access**: [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

2. **Edge et al. (2024)**
   - **Title**: From Local to Global: A Graph RAG Approach to Query-Focused Summarization
   - **Authors**: Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Jonathan Larson
   - **Venue**: arXiv:2404.16130
   - **Expected filename**: `data/papers/edge_2024_graphrag.pdf`
   - **Open Access**: [https://arxiv.org/abs/2404.16130](https://arxiv.org/abs/2404.16130)

3. **Komarov et al. (2026)**
   - **Title**: RAGU: A Multi-Step GraphRAG Engine with a Compact Domain-Adapted LLM
   - **Authors**: Mikhail Komarov, Ivan Bondarenko, Stanislav Shtuka, Oleg Sedukhin, Roman Shuvalov, Yana Dementyeva, Matvey Solovyov, Nikolay O. Nikitin
   - **Venue**: EMNLP 2026 System Demonstrations / arXiv:2607.11683
   - **Expected filename**: `data/papers/komarov_2026_ragu.pdf`
   - **Open Access**: [https://arxiv.org/abs/2607.11683](https://arxiv.org/abs/2607.11683)

## File Placement & Verification

Place the downloaded PDF files directly into `data/papers/` with the exact filenames
listed above.

Verify presence and readability with:

```powershell
python -c "from pathlib import Path; from pypdf import PdfReader; [print(f'{p.name}: {len(PdfReader(p).pages)} pages') for p in Path('data/papers').glob('*.pdf')]"
```

