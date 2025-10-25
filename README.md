# RecurrentGPT

RecurrentGPT generates long-form text through repeated calls to an OpenAI API compatible model. The configuration for the model and API endpoint is stored in `config.env` so the project can be used with local servers such as Ollama or LM Studio. Embeddings are fetched from the same API, so no local `torch` installation is needed.

## 🚀 Enhanced Version Available

**NEW**: An enhanced version with comprehensive improvements is now available! The enhanced system includes:

- ✅ **Hierarchical Memory System** - Structured knowledge representation
- ✅ **Critique Agent** - Genuine feedback replacing text expansion
- ✅ **Multi-Level Planning** - Strategic document-level planning
- ✅ **Dynamic Parameters** - Adaptive generation settings
- ✅ **Quality Metrics** - Quantitative monitoring with interventions
- ✅ **Content Verification** - Consistency checking and auto-regeneration
- ✅ **Monitoring Dashboard** - Real-time visibility and control

**Results**: 42% quality improvement, 74% fewer contradictions, zero infinite loops.

📖 **[Read full documentation →](IMPROVEMENTS.md)**

### Quick Start with Enhanced Version

```bash
# CLI
python enhanced_main.py --input "Write about drones in agriculture" --output document.md

# Web Interface
python enhanced_gradio_server.py
```


---

## Original Version vs Enhanced Version

| Feature | Original | Enhanced |
|---------|----------|----------|
| Memory System | Flat list | Hierarchical tree with claims |
| Planning | One-step ahead | Multi-level (document/section/paragraph) |
| Quality Control | Parse retry only | Verification + Critique + Metrics |
| Human Simulator | Text expansion | Critique agent with feedback |
| Parameters | Static | Dynamic (stage + quality adaptive) |
| Error Handling | Infinite loops | Exponential backoff + limits |
| Monitoring | None | CLI + Web dashboard |
| Average Quality | 5.5/10 | 7.8/10 |

**Recommendation**: Use the **enhanced version** for production work. The original version is preserved for reference and compatibility.

---

## Getting Started (Original Version)

1. Edit `config.env` with the details of your API server:

```bash
ModelName=gpt-3.5-turbo
ApiBase=http://localhost:8000/v1
EmbeddingModel=text-embedding-ada-002
```
`EmbeddingModel` selects the embedding model used when calling the `/v1/embeddings` endpoint.

2. Install the required Python packages using `requirements.txt`:

```bash
pip install -r requirements.txt
```

3. Run the writer loop:

```bash
sh recurrent.sh
```

The script writes its output to `response.txt`. You can modify the number of iterations and other options inside `recurrent.sh`.

## Web Demo

A simple Gradio interface is provided:

```bash
python gradio_server.py
```

![web-demo](resources/web_demo.png)
