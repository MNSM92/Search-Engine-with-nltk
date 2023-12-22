from operator import itemgetter
from tools import *

documents, queries, mappings = load_data()
doc_words, qry_words = {doc_id: get_words(d) for doc_id, d in documents.items()}, {qry_id: get_words(q) for qry_id, q in queries.items()}
doc_terms, qry_terms = {doc_id: get_terms(d) for doc_id, d in documents.items()}, {qry_id: get_terms(q) for qry_id, q in queries.items()}
all_terms = collect_vocabulary(doc_terms, qry_terms)
doc_vectors, qry_vectors = vectorize(doc_terms, all_terms), vectorize(qry_terms, all_terms)
doc_idfs = calculate_idfs(all_terms, doc_terms)
doc_vectors = vectorize_idf(doc_terms, doc_idfs, all_terms)


precision_all = 0.0
for query_id in mappings.keys():
    gold_standard = mappings.get(str(query_id))
    query = qry_vectors.get(str(query_id))
    result = ""
    model_output = []
    max_sim = 0.0
    prefiltered_docs = prefilter(doc_terms, qry_terms.get(str(query_id)))
    for doc_id in prefiltered_docs:
        document = doc_vectors.get(doc_id)
        cosine = calculate_cosine(query, document)
        if cosine >= max_sim:
            max_sim = cosine
            result = doc_id
    model_output.append(result)
    precision = calculate_precision(model_output, gold_standard)
    print(f"{str(query_id)}: {str(precision)}")
    precision_all += precision

print(precision_all/len(mappings.keys()))

rank_all = 0.0
for query_id in mappings.keys():
    gold_standard = mappings.get(str(query_id))
    query = qry_vectors.get(str(query_id))
    results = {}
    for doc_id in doc_vectors.keys():
        document = doc_vectors.get(doc_id)
        cosine = calculate_cosine(query, document)
        results[doc_id] = cosine
    sorted_results = sorted(results.items(), key=itemgetter(1), reverse=True)
    index = 0
    found = False
    while found == False:
        item = sorted_results[index]
        index += 1
        if index == len(sorted_results):
            found = True
        if item[0] in gold_standard:
            found = True
            print(f"{str(query_id)}: {str(float(1) / float(index))}")
            rank_all += float(1) / float(index)

print(rank_all / float(len(mappings.keys())))
