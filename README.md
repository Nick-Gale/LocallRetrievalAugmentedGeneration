# Retrieval Augmented Generation

At its core, the Retrieval Augmented Generation (RAG) idea is very simple --- embded into the context window of a Large Language Model (LLM) a tensor of information that you wish to you operate on. The utility is derived not from a technical standpoint, the model continues to function much like a next-word predictor as before, but from the ability of the user to guide a models responses into a domain specific area and retrieve information sourcing for the information the model provides. This serves two critical functions: the ability to check the model veracity (or at least explain to an expert the information source from which a statement is derived) and streamlining the model into expert domains with trusted information.

## Mechanics

The core mechanic of the RAG works in the following way:

1. Convert data into a unified format.
2. Chunk data into particular sizes (an important hyperparameter).
3. Take a series of input tokens and construct a query vector.
4. Compute over all chunks a similarity metric (typically cosine), and return the k-th most similar.
5. Integrate these k vectors into the response from the LLM.

In practice we use two models: one to create embeddings or chunks and another to respond to the queries. The similarity metric is chosen to be the cosine projection or dot product --- a simple measure of how much one vector projects onto other --- in principle this is arbitrary but it works well for most use cases and changing it would be a matter of active research. The key hyperparameters a practitioner should track are: chunk size, chunk overlap, and chunk retrieval. These represent how much of a document is retrieved and how much information is shared between components of document. Note: this is *not* the state-of-the-art for RAG techniques and indeed represents a massive over-simplification. For instructive technical documentation please refer to https://github.com/NirDiamant/RAG_Techniques.

## Implementation

The vast suite of public AI tools makes implementation of a local RAG quite straightforward. The technical stacks is as follows:
1. pypdf --- useful to parse pdfs into Python.
2. chromaDB --- a good general purpose database manager.
3. Ollama --- serving as an LLM API endpoint.
4. llama-index-(core,embeddings-ollama,llms-ollama,vector-stores-chroma) --- a suite of utilities to create embeddings, models, and data interactions.

## Testing
