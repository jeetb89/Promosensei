# embedding_utils.py

import os
import logging
from openai import OpenAI
from typing import List
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found in environment variables.")
    raise EnvironmentError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

def embed_text(text: str) -> List[float]:
    """
    Generate an embedding vector for the input text using OpenAI embeddings.
    """
    try:
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logger.error(f"Failed to embed text: {e}")
        raise RuntimeError("Embedding generation failed") from e
