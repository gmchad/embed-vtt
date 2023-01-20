# Embed-VTT
This repo uses openai embeddings and the [pinecone](https://www.pinecone.io/) vector database to
generate and query embeddings from a VTT file.

The purpose of this repo was to implement semantic search to provide an extra resource
in understanding Andrej Karpathy's latest video: [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY&), 
however it is general enough to use for any transcript

shoutout to miguel's [yt-whisper](https://github.com/m1guelpf/yt-whisper) library for helping with the youtube
transcription. The `data/` in this repo was generated using the `small` model

## Setup
### Install
```
pip install -r requirements.txt
```
### Environment
```
cp .env.sample .env
```
you'll need an API keys from [openai](https://beta.openai.com/) & [pinecone](https://www.pinecone.io/)   
`OPENAI_KEY=***`    
`PINECONE_KEY=***`    
### Pinecone
Head over to [pinecone](https://www.pinecone.io/) and create an index with dimension **1536**

## Usage
### Generate Embeddings from VTT file
this will save an embedding csv file as `{file_name}_embeddings.csv`
```
python embed_vtt.py generate --vtt-file="data/karpathy.vtt"
```
### Upload Embeddings from a CSV Embedding file
```
python embed_vtt.py upload --csv-embedding-file="data/karpathy_embeddings.csv"
```
### Query Embeddings from text
```
python embed_vtt.py query --text="the usefulness of trill tensors"
```
sample output
```
0.81: But let me talk through it. It uses softmax. So trill here is this matrix, lower triangular ones. 00:54:52.240-00:55:01.440
0.81: but torches this function called trill, which is short for a triangular, something like that. 00:48:48.960-00:48:55.920
0.80: which is a very thin wrapper around basically a tensor of shape vocab size by vocab size. 00:23:17.920-00:23:23.280
0.79: I'm creating this trill variable. Trill is not a parameter of the module. So in sort of pytorch 01:19:36.880-01:19:42.160
0.79: does that. And I'm going to start to use the PyTorch library, and specifically the Torch.tensor 00:12:54.320-00:12:59.200
```

## License
This script is open-source and licensed under the MIT License.