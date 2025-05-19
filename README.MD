

```
ollama pull qwen3:8b
ollama pull nomic-embed-text
docker run -p 6333:6333 -p 6334:6334 -v "$(pwd)/qdrant_storage:/qdrant/storage:z" qdrant/qdrant
```
