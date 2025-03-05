from haystack import Pipeline
from haystack_integrations.components.embedders.ollama import OllamaDocumentEmbedder

from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import NLTKDocumentSplitter, DocumentCleaner
from haystack.components.writers import DocumentWriter
from milvus_haystack import MilvusDocumentStore


path = "../data/youfile.pdf"


#  INDEXING  #


### Initialize the DB store
milvus_store = MilvusDocumentStore(
    connection_args={"uri": "http://localhost:19530/"},
    collection_name="Rag_AI_agent",
)

### Initialize the Components
pdf_converter = PyPDFToDocument()

cleaner = DocumentCleaner(
    remove_empty_lines=True,
    remove_extra_whitespaces=True,
    remove_repeated_substrings=False,
)
splitter = NLTKDocumentSplitter(
    split_by="sentence", split_length=256, split_overlap=1, split_threshold=1
)

embedder = OllamaDocumentEmbedder(model="llama3.2:3b", url="http://localhost:11434/")
writer = DocumentWriter(document_store=milvus_store)

### Add components to Pipeline
p_indexing = Pipeline()
p_indexing.add_component("converter", pdf_converter)
p_indexing.add_component("cleaner", cleaner)
p_indexing.add_component("embedder", embedder)
p_indexing.add_component("splitter", splitter)
p_indexing.add_component("writer", writer)

### Connect components in Pipeline
p_indexing.connect("converter", "cleaner")
p_indexing.connect("cleaner", "splitter")
p_indexing.connect("splitter", "embedder")
p_indexing.connect("embedder", "writer")

p_indexing.run({"converter": {"sources": [path]}})
