from simplemma import text_lemmatizer
from haystack.schema import Document
from typing import Optional, List, Dict, Union, Any
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

    def run(self, documents: List[Document]) -> List[Document]:

        ldocuments = []
        for doc in documents:
            
            doc_words = text_lemmatizer(doc.content, lang=self.base_lang)
            lemmatized_doc = ' '.join(doc_words)

            # wnl = WordNetLemmatizer()
            # doc_words = nltk.word_tokenize(doc.content)
            # lemmatized_doc = ' '.join([wnl.lemmatize(words) for words in doc_words])
            
            doc.content = lemmatized_doc
            ldocuments.append(doc)

        return {"documents": ldocuments}, "output_1"
    
    def run_batch(self, documents: list, **kwargs):
        return