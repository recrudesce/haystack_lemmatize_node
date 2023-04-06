# haystack_lemmatize_node

```python
import logging
import re

from datasets import load_dataset
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser, BM25Retriever
from haystack.pipelines import Pipeline
from haystack_lemmatize_node import LemmatizeDocuments


logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

document_store = InMemoryDocumentStore(use_bm25=True)

dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
document_store.write_documents(dataset)

retriever = BM25Retriever(document_store=document_store, top_k=2)

lfqa_prompt = PromptTemplate(
    name="lfqa",
    prompt_text="Given the context please answer the question using your own words. Generate a comprehensive, summarized answer. If the information is not included in the provided context, reply with 'Provided documents didn't contain the necessary information to provide the answer'\n\nContext: {documents}\n\nQuestion: {query} \n\nAnswer:",
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(
    model_name_or_path="text-davinci-003",
    default_prompt_template=lfqa_prompt,
    max_length=500,
    api_key="sk-OPENAIKEY",
)

lemmatize = LemmatizeDocuments() # you can pass the `base_lang=XX` argument here too, where XX is a language as listed here: https://pypi.org/project/simplemma/

pipe = Pipeline()
pipe.add_node(component=retriever, name="Retriever", inputs=["Query"])
pipe.add_node(component=lemmatize, name="Lemmatize", inputs=["Retriever"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["Lemmatize"])

query = "What does the Rhodes Statue look like?"
  
output = pipe.run(query)

print(output['answers'][0].answer)
```
