# === src/qa_chain.py ===
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever

def setup_qa_chain(llm, db_main, db_memory, prompt):



    # Combine main + memory
    retriever = EnsembleRetriever(
        retrievers=[
            db_main.as_retriever(search_kwargs={"k": 5}),
            db_memory.as_retriever(search_kwargs={"k": 3}),
        ],
        weights=[0.7, 1.3]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt},
    )