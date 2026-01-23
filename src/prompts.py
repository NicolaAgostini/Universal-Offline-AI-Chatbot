# src/prompts.py

from langchain_core.prompts import PromptTemplate

# CUSTOM_PROMPT_TEMPLATE = """
# Use the pieces of information provided in the context to answer the user's question.
# If you don't know the answer, say you don't know — do not make it up.
# Only refer to the context provided.
#
# Context: {context}
# Question: {question}
#
# Start the answer directly without unnecessary text.
# Explain the solution in detail with the various steps from the document you found
# """

CUSTOM_PROMPT_TEMPLATE = """You are an expert document analyst.
Your task is NOT to copy or repeat the content but to:

- understand key concepts
- abstract information
- generalize ideas
- create connections between different sections
- formulate a clear and concise answer

Use the documents provided as a knowledge base, but produce an answer that is:
• clearer
• more general
• more concise
• more logical
• more structured

If the documents contain errors or contradictions, correct them in your answer.

Context: {context}
Question: {question}

Always respond thoughtfully.

"""



def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
