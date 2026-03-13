"""Build index using a remote embedding server (e.g. RunAI inference job).

The embedding server must expose POST /embed and GET /info endpoints
(see scripts/embedding_server.py).

Usage:
    python vendor/KohakuRAG/scripts/wattbot_build_index.py \
        --config vendor/KohakuRAG/configs/jinav4/remote_index.py

    # Or override the URL via environment variable:
    EMBEDDING_SERVICE_URL=http://my-embedding-server:8080 \
        python vendor/KohakuRAG/scripts/wattbot_build_index.py \
        --config vendor/KohakuRAG/configs/jinav4/remote_index.py
"""

# Document and database settings
metadata = "../../data/metadata.csv"
docs_dir = "../../data/corpus"
db = "../../data/embeddings/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
use_citations = False

# Use remote embedding server instead of loading model locally
embedding_model = "remote"
embedding_service_url = "http://localhost:8080"  # Override via EMBEDDING_SERVICE_URL env var

# Paragraph embedding mode
paragraph_embedding_mode = "averaged"
