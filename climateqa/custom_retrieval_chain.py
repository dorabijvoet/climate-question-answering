from __future__ import annotations
import inspect
from typing import Any, Dict, List, Optional

from pydantic import Extra

from langchain.schema.language_model import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate

from typing import Any, Dict, List

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.qa_with_sources.base import BaseQAWithSourcesChain
from langchain.docstore.document import Document
from langchain.pydantic_v1 import Field
from langchain.schema import BaseRetriever

from langchain.chains import RetrievalQAWithSourcesChain


from langchain.chains.router.llm_router import LLMRouterChain

class CustomRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):

    fallback_answer:str = "No sources available to answer this question."

    def _call(self,inputs,run_manager=None):
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        accepts_run_manager = (
            "run_manager" in inspect.signature(self._get_docs).parameters
        )
        if accepts_run_manager:
            docs = self._get_docs(inputs, run_manager=_run_manager)
        else:
            docs = self._get_docs(inputs)  # type: ignore[call-arg]

        
        if len(docs) == 0:
            answer = self.fallback_answer
            sources = []
        else:

            answer = self.combine_documents_chain.run(
                input_documents=docs, callbacks=_run_manager.get_child(), **inputs
            )
            answer, sources = self._split_sources(answer)

        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: sources,
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result
