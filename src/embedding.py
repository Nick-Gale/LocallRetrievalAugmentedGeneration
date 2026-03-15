import pathlib, llama_index, pickle

def _create_indexed_data(data_path: str, emdedding_model):
    """Read files from a data source and index them according to an embedding model."""

    assert pathlib.Path(data_path).exists(), "Please enter a valid data path."

    data = llama_index.core.SimpleDirectoryReader(data_path).load_data()

    index = llama_index.core.from_documents(data, embed_model=embedding_model)

    return index


def _load_indexed_data(index_path: str):
    """Loads the pickled embeddings stored in the local data directory."""
    
    assert pathlib.Path(index_path).exists(), "Please enter a valid data path."
    
    with open(index_path, 'rb') as file:
        index = pickle.load(file)

    return index


def _pickle_indexed_data(index_path:str, index):
    """Store index data in serialised format."""

    if pathlib.Path(index_path).exists():
        print("[INFO] overwriting existing index.")
    else:
        print("[INFO] saving first indexation of data.")

    with open(index_path, 'wb') as file:
        pickle.dump(index, file)

    return None

def indexation(index_path: str, emedding_model, data_path="", reconstruct=False):
    """Create or load the RAG indexation according to an embedding model."""
    
    exisiting_data = pathlib.Path(index_path).exists()
    
    if not reconstruct and existing_data:
        index = _load_index_data(index_path)
    else:
        index = _create_indexed_data(data_path, emedding_model)
        _pickle_indexed_data(index_path, index)

    return index

