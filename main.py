import os
from pathlib import Path

import yaml

from paperqa.contrib import ZoteroDB
from paperqa.docs import Docs, OpenAIEmbeddingModel, OpenAILLMModel
from paperqa.utils import load_embeddings, save_embeddings
from utils import skip_run

with open("./config.yml") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
    # Configure API keys
    os.environ["OPENAI_API_KEY"] = config["openai_api_key"]

with skip_run("skip", "initial_testing") as check, check():
    # Zotero library
    zotero_storage_path = Path(config["zotero"]["storage_path"])
    zotero = ZoteroDB(
        library_id=config["zotero"]["user_id"],
        api_key=config["zotero"]["api_key"],
        storage=zotero_storage_path,
    )
    collection = zotero.iterate(limit=1, collection_name="Google")

    # Add LLM
    openai_embedding_model = OpenAIEmbeddingModel()
    openai_llm_model = OpenAILLMModel(
        config={"model": "gpt-3.5-turbo", "temperature": 0.1, "frequency_penalty": 1.5}
    )

    # Create docs object
    docs = Docs(llm_model=openai_llm_model, name="test", index_path="./indexes/")

    # Add each item to doc
    for item in collection:
        docs.add(item.pdf, docname=item.key)

    answer = docs.query(
        "Write an introduction on the transferability of features in deep neural network"
    )
    save_embeddings(docs, index_file_path="./indexes/docs.pkl")
    print(answer)

with skip_run("skip", "testing_saving_and_loading") as check, check():
    docs = load_embeddings("./indexes/docs.pkl")
    docs.set_client()
    answer = docs.query(
        "Write an introduction on the transferability of features in deep neural network"
    )
    print(answer)
