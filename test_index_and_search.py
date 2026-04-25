from main import vectorstore, RERANKER

# Tiny sample documents to index
texts = [
    "Our refund policy states that refunds are processed within 30 days of purchase.",
    "Shipping usually takes 3-5 business days and is handled by our carrier.",
    "To request a refund, contact support@example.com with your order id.",
    "We do not accept returns of opened software products.",
]
metadatas = [
    {"source_file": "sample1.txt", "chunk_number": 1},
    {"source_file": "sample2.txt", "chunk_number": 1},
    {"source_file": "sample3.txt", "chunk_number": 1},
    {"source_file": "sample4.txt", "chunk_number": 1},
]

print('Adding sample texts to vectorstore...')
vectorstore.add_texts(texts=texts, metadatas=metadatas)

query = 'How do I get a refund?'
print('Running similarity_search_with_score...')
try:
    results = vectorstore.similarity_search_with_score(query, k=5)
except Exception as e:
    print('similarity_search_with_score failed:', e)
    results = []

print('Raw results:')
for i, (doc, score) in enumerate(results):
    print(i+1, 'score=', score, 'source=', doc.metadata.get('source_file'), 'text=', doc.page_content[:120])

# If reranker exists, show reranker scores
if RERANKER is not None and results:
    pairs = [(query, doc.page_content) for doc, _ in results]
    try:
        rerank_scores = RERANKER.predict(pairs)
        print('Rerank scores:', rerank_scores)
    except Exception as e:
        print('Reranker predict failed:', e)
