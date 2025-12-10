# === src/qa_chain.py ===
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

def setup_qa_chain(llm, db, base_prompt):

    italian_wrapper = """
Use the pieces of information provided in the context to answer the user's question.
If you don't know the answer, say you don't know — do not make it up.
Only refer to the context provided.

Context: {context}
Question: {question}

Start the answer directly without unnecessary text.
    """

    prompt = PromptTemplate(
        template=italian_wrapper,
        input_variables=["context", "query"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )