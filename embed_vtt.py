import pinecone
import webvtt
import openai
from openai.embeddings_utils import get_embedding
import pandas as pd
import tiktoken
import random
from ast import literal_eval
import itertools

import os
from dotenv import load_dotenv
load_dotenv()

# read key from .env file
openai.api_key = os.getenv('OPENAI_KEY')

def read_data():
    captions = webvtt.read('gpt_tutorial.vtt')
    text = [caption.text for caption in captions]
    df = pd.DataFrame(text, columns=['captions'])
    return df

def get_encodings(df):
    embedding_encoding = "cl100k_base"  # this the encoding for text-embedding-ada-002

    encoding = tiktoken.get_encoding(embedding_encoding)
    df["n_tokens"] = df["captions"].apply(lambda x: len(encoding.encode(x)))

    return df

def get_embeddings(df):
    embedding_model = "text-embedding-ada-002"
    df["embedding"] = df["captions"].apply(
        lambda x: get_embedding(x, engine=embedding_model))
    
    df.to_csv("gpt_tutorial_embeds.csv")

def generate_embeddings():
    df = read_data()
    encodings = get_encodings(df)
    embeddings = get_embeddings(df)

######

def query_embeddings(query, index):
    xq = openai.Embedding.create(input=query, engine="text-embedding-ada-002")['data'][0]['embedding']
    # query, returning the top 5 most similar results
    res = index.query([xq], top_k=5, include_metadata=True)
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata']['captions']}")

def chunks(iterable, batch_size=100):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def upload_embeddings(index):
    df = pd.read_csv("gpt_tutorial_embeds.csv")

    df["embedding"] = df["embedding"].apply(lambda x: literal_eval(x))
    example_data_generator = map(lambda i: (
        f'id-{i}', df["embedding"][i], {"captions": df["captions"][i]}), range(len(df["embedding"])))

    for ids_vectors_chunk in chunks(example_data_generator, batch_size=100):
        print("uploading chunk")
        index.upsert(vectors=ids_vectors_chunk)

def init_pinecone():
    pinecone_key = os.getenv('PINECONE_KEY')
    pinecone.init(api_key=pinecone_key,
                environment="us-west1-gcp")
    index = pinecone.Index(pinecone.list_indexes()[0])
    return index

def main():
    index = init_pinecone()
    #upload_embeddings(index)
    query_embeddings("the usefulness of tril tensors", index)

if __name__ == '__main__':
    df = None
    main()