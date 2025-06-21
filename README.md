# RecurrentGPT

RecurrentGPT generates long-form text through repeated calls to an OpenAI API compatible model. The configuration for the model and API endpoint is stored in `config.env` so the project can be used with local servers such as Ollama or LM Studio. Embeddings are fetched from the same API, so no local `torch` installation is needed.


## Getting Started

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
