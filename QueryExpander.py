from openai import OpenAI
import os
import json
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

MULTI_QUERY_PROMPT = """You are a query reformulation assistant for a chess knowledge 
retrieval system backed by chess books.

Given a user's question, generate {n} alternative versions of it that would help 
retrieve relevant passages. Vary them by:
- Expanding abbreviations or notation into full terms
- Rephrasing using chess synonyms or related terminology  
- Breaking it into a narrower, more specific sub-question if applicable

Do not answer the question. Only output reformulated queries.

Original question: {question}

Respond ONLY in this JSON format, nothing else:
{{"queries": ["query1", "query2", "query3"]}}
"""

def generate_query_variants(question, n=3):
    prompt = MULTI_QUERY_PROMPT.format(question=question, n=n)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=5000
    )
    result = json.loads(response.choices[0].message.content)
    variants = result["queries"]
    return [question] + variants  # keep original in the mix too