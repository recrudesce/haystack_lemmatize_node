from simplemma import text_lemmatizer
from haystack.schema import Document
from typing import Optional, List, Dict, Tuple
from haystack.nodes.base import BaseComponent
from typing import Optional


class LemmatizeDocuments(BaseComponent):
    def __init__(
        self,
        base_lang: Optional[str] = "en",
    ):
        super().__init__()
        self.base_lang = base_lang

    outgoing_edges = 1

    def run(self, documents: List[Document]) -> Tuple[Dict, str]:
        ldocuments = []
        for doc in documents:            
            doc_words = text_lemmatizer(doc.content, lang=self.base_lang)
            doc.content = ' '.join(doc_words)
            ldocuments.append(doc)

        return {"documents": ldocuments}, "output_1"
    
    def run_batch(self, documents: list, **kwargs):
        return