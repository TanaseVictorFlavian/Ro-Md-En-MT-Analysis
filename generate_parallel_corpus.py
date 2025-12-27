import os
from xml.parsers.expat import model
from google import genai
from google.api_core import exceptions
import asyncio
import time
import dotenv
from pathlib import Path
import json
import time

dotenv.load_dotenv()

system_prompt = "You are an expert translator specializing in Romanian/Moldovan to English"

prompt = """
You will translate Romanian and Moldovan text into English while preserving the original meaning and context. 
Ensure that the translations are accurate, culturally appropriate, and maintain the nuances of the source language.

Input_format: 
    <text>target_sentence1</text>
    <text>target_sentence2</text>
    ...
    <text>target_sentenceN</text>

Output_format: 
    <text>translated_sentence1</text>
    <text>translated_sentence2</text>e
    ...
    <text>translated_sentenceN</text>

Respect the input and output formats strictly. 

Target sentences: 

{sentences}
"""

REQUEST_LIMIT = 20
PAIRS_PER_REQUEST = 200

def strip_target(target):
    return target.replace("<text>", "").replace("</text>", "").strip()

def write_pair(source, target, index): 
    file_name = f"{index:05}.json"
    pair = {"source": source, "target": strip_target(target)}

    with open(parallel_corpus_dir / file_name, "w", encoding="utf-8") as f:
        json.dump(pair, f, ensure_ascii=False)

def generate_pairs(source_data:list[str], pair_idx_start: int, client, pairs_per_request: int = PAIRS_PER_REQUEST):
    source = source_data[pair_idx_start : pair_idx_start + pairs_per_request]
    
    if not source:
        print("Finished!")
        return 
    
    formatted_source = "\n".join(f"<text>{sentence}</text>" for sentence in source)
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt.format(sentences=formatted_source))
        targets = [strip_target(t) for t in response.text.splitlines()]

        if len(targets) != len(source):
            raise ValueError("Source/target length mismatch")
        
        for i, (s, t) in enumerate(zip(source,targets)):
            write_pair(s, t, pair_idx_start + i + 1)

    except Exception as e: 
        print(f"Request limit reached: {e}")

parallel_corpus_dir = Path.cwd() / "parallel_corpus"

if not os.path.exists(parallel_corpus_dir):
    os.makedirs(parallel_corpus_dir)

with open("ro_source.txt", "r", encoding="utf-8") as f:
    ro_sentences = f.read().splitlines()

with open("md_source.txt", "r", encoding="utf-8") as f:
    md_sentences = f.read().splitlines()

merged_dataset = ro_sentences + md_sentences

print(len(merged_dataset))
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY2"))

for _ in range(REQUEST_LIMIT):
    # Time tracking to respect the request per minute limit
    start_time = time.time()
    pair_idx = len(os.listdir(parallel_corpus_dir))

    print(f"Pair index:{pair_idx}")

    generate_pairs(merged_dataset, pair_idx, client)
    elapsed = time.time() - start_time
    time_to_wait = 13 - elapsed

    if time_to_wait > 0:
        time.sleep(time_to_wait)
    