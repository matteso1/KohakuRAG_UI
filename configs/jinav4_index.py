"""Build JinaV4 index for KohakuRAG_UI.

Usage:
    cd KohakuRAG && kogine run scripts/wattbot_build_index.py --config ../configs/jinav4_index.py
"""

from kohakuengine import Config

# Document and database settings - paths relative to KohakuRAG subfolder
metadata = "../data/metadata.csv"
docs_dir = "../artifacts/docs"
db = "../artifacts/wattbot_jinav4.db"
table_prefix = "wattbot_jv4"
use_citations = False

# JinaV4 embedding settings (matching winning config)
embedding_model = "jinav4"
embedding_dim = 512  # Matching ensemble runner config
embedding_task = "retrieval"

# Paragraph embedding mode - "both" allows runtime toggle
paragraph_embedding_mode = "both"


def config_gen():
    return Config.from_globals()
