from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
from milvus_haystack.milvus_embedding_retriever import MilvusEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from milvus_haystack import MilvusDocumentStore
from haystack import Pipeline

milvus_store = MilvusDocumentStore(
    connection_args={"uri": "http://localhost:19530/"},
    collection_name="Rag_AI_agent",
)


#  RAG System  #
def generator_llm(model, temperature, url):
    """
        Creates an instance of OllamaGenerator with the specified model, temperature, and URL.
        Args:
            model (str): The name or path of the model to be used by the generator.
            temperature (float): The temperature parameter for the generation process, controlling the randomness of the
    output.
            url (str): The URL endpoint for the generator.
        Returns:
            OllamaGenerator: An instance of OllamaGenerator configured with the provided model, temperature, and URL.
    """

    generator = OllamaGenerator(
        model=model, url=url, generation_kwargs={"temperature": temperature}
    )
    return generator


prompt_template = """
Using only the information contained in these document returnn a brief answer (max 150 words).
If teh answer cannot be inferred from the documents, respond \"I don't know."\
Documents:
{%for doc in documents%}
    {{doc.content}}
{% endfor %}
Question: {{question}}
Answer:
"""

generator = generator_llm(
    model="phi3.5:3.8b", temperature=0.9, url="http://localhost:11434/"
)
### Used to embed the text for the query input
text_embedder = OllamaTextEmbedder(model="llama3.2:3b")
retriever = MilvusEmbeddingRetriever(document_store=milvus_store, top_k=5)
prompt_builder = PromptBuilder(
    template=prompt_template,
)
rag_pipeline = Pipeline()
rag_pipeline.add_component("retriever", retriever)
rag_pipeline.add_component("generator", generator)
rag_pipeline.add_component("text_embedder", text_embedder)
rag_pipeline.add_component("prompt_builder", prompt_builder)

rag_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
rag_pipeline.connect("prompt_builder", "generator")

question = str(input("Enter your question: \n"))

results = rag_pipeline.run(
    {
        "text_embedder": {"text": question},
        "prompt_builder": {"question": question},
    }
)

for res in results["generator"]["replies"]:
    print("\nthe answer is: \n", res)
