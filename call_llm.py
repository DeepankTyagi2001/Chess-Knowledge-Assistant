from openai import OpenAI

import os 
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

def call_llm(query,context):
  prompt = f"""
  You are a RAG assistant.

  Answer ONLY using the context below.

  If the answer is not present in the context, reply exactly:

  "I don't know based on the provided documents."

  Context:
  {context}

  Question:
  {query}
  """
  client= OpenAI(
    api_key=os.getenv("OPEN_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
  )
  response = client.chat.completions.create(
      model="gpt-4o-mini",
      messages=[{"role": "user", "content": prompt}]
  )
  return (response.choices[0].message.content)