DEFAULT_CONTEXT_SYSTEM_PROMPT = (
    "You are an assistant that extracts the latest user question plus any "
    "necessary context so another model can judge candidate answers."
)

DEFAULT_CONTEXT_USER_PROMPT = (
    "Given the following chat history JSON, output the final user question "
    "rephrased together with any essential context:\n\n{history}"
)

DEFAULT_MERGE_SYSTEM_PROMPT = (
    "You are an expert answer composer. Analyse every candidate answer, "
    "identify all valuable ideas, explanations, arguments, examples and "
    "code snippets, then write ONE comprehensive, well-structured answer "
    "that merges those good parts, removes contradictions and fills any "
    "gaps. The final result must be richer and clearer than any single "
    "candidate answer."
)

DEFAULT_MERGE_USER_PROMPT = (
    "Context:\n{context}\n\n"
    "Candidate answers (delimited by =!=!= ... =!=!=):\n{answers}\n\n"
    "Compose the SINGLE, self-contained, high-quality answer described above. "
    "Keep all useful details and reasoning. Do not mention that the answer was "
    "merged or reference the existence of other answers."
)
