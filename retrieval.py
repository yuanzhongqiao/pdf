# retrieval.py
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
import torch
import logging
from config import TOP_K

# Logging setup
logger = logging.getLogger(__name__)

def find_most_relevant_context_faiss(contexts, question, embeddings, tokenizer, model):
    question_embedding = get_embeddings([question], tokenizer, model)[0]
    context_embeddings = embeddings

    dimension = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(context_embeddings)

    _, indices = index.search(question_embedding.reshape(1, -1), min(TOP_K, len(context_embeddings)))
    return [contexts[idx] for idx in indices[0] if idx < len(contexts)]

def find_most_relevant_context_bm25(contexts, question):
    tokenized_corpus = [doc.split() for doc in contexts]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_question = question.split()
    return bm25.get_top_n(tokenized_question, contexts, n=min(TOP_K, len(contexts)))

def rerank_results(contexts, question, tokenizer, model):
    inputs = [f"Query: {question} Document: {context}" for context in contexts]
    inputs_tokenized = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embeddings = model(**inputs_tokenized).pooler_output.cpu().numpy()

    query_embedding = embeddings[0]
    context_embeddings = embeddings[1:]
    scores = cosine_similarity([query_embedding], context_embeddings)[0]

    # Boost scores for contexts with question keyword overlap
    question_words = set(question.lower().split())
    for i, context in enumerate(contexts):
        context_words = set(context.lower().split())
        overlap = len(question_words & context_words) / len(question_words)
        scores[i] += overlap * 0.2  # Weight keyword overlap

    ranked_contexts = sorted(zip(contexts, scores), key=lambda x: x[1], reverse=True)
    return [context for context, _ in ranked_contexts][:TOP_K]

def hybrid_search(contexts, question, embeddings, tokenizer_embed, model_embed, tokenizer_rerank, model_rerank):
    faiss_results = find_most_relevant_context_faiss(contexts, question, embeddings, tokenizer_embed, model_embed)
    bm25_results = find_most_relevant_context_bm25(contexts, question)

    combined_scores = {}
    for ctx in faiss_results:
        combined_scores[ctx] = combined_scores.get(ctx, 0) + 0.6
    for ctx in bm25_results:
        combined_scores[ctx] = combined_scores.get(ctx, 0) + 0.4

    combined_results = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
    if not combined_results:
        return ["I don't know."]
    
    return rerank_results(combined_results, question, tokenizer_rerank, model_rerank)

def get_embeddings(texts, tokenizer, model, batch_size=32):
    if not texts:
        logger.warning("Empty text list provided to get_embeddings.")
        return np.array([])

    valid_texts = [str(text).strip() for text in texts if text is not None and str(text).strip()]
    if not valid_texts:
        logger.warning("No valid texts after filtering.")
        return np.array([])

    embeddings = []
    for i in range(0, len(valid_texts), batch_size):
        batch = valid_texts[i:i + batch_size]
        try:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings.append(outputs.pooler_output.cpu().numpy())
        except Exception as e:
            logger.error(f"Error processing batch {i//batch_size}: {str(e)}")

    if not embeddings:
        logger.warning("No embeddings generated.")
        return np.array([])

    return np.concatenate(embeddings, axis=0)