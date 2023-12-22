import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize, WordNetLemmatizer, pos_tag
from nltk.stem.lancaster import LancasterStemmer
import string, math
nltk.download('stopwords')


def read_documents():
    f = open("cisi/CISI.ALL")
    merged = ""

    for a_line in f.readlines():
        if a_line.startswith("."):
            merged += "\n" + a_line.strip()
        else:
            merged += " " + a_line.strip()

    documents = {}

    content = ""
    doc_id = ""

    for a_line in merged.split("\n"):
        if a_line.startswith(".I"):
            doc_id = a_line.split(" ")[1].strip()
        elif a_line.startswith(".X"):
            documents[doc_id] = content
            content = ""
            doc_id = ""
        else:
            content += a_line.strip()[3:] + " "
    f.close()
    return documents


def read_queries():
    f = open("cisi/CISI.QRY")
    merged = ""

    for a_line in f.readlines():
        if a_line.startswith("."):
            merged += "\n" + a_line.strip()
        else:
            merged += " " + a_line.strip()

    queries = {}

    content = ""
    qry_id = ""

    for a_line in merged.split("\n"):
        if a_line.startswith(".I"):
            if not content == "":
                queries[qry_id] = content
                content = ""
                qry_id = ""
            qry_id = a_line.split(" ")[1].strip()
        elif a_line.startswith(".W") or a_line.startswith(".T"):
            content += a_line.strip()[3:] + " "
    queries[qry_id] = content
    f.close()
    return queries


def read_mappings():
    f = open("cisi/CISI.REL")

    mappings = {}

    for a_line in f.readlines():
        voc = a_line.strip().split()
        key = voc[0].strip()
        current_value = voc[1].strip()
        value = []
        if key in mappings.keys():
            value = mappings.get(key)
        value.append(current_value)
        mappings[key] = value

    f.close()
    return mappings


def get_words(text):
    word_list = [word for word in word_tokenize(text.lower())]
    return word_list


def retrieve_documents(doc_words, query):
    docs = []
    for doc_id in doc_words.keys():
        found = False
        i = 0
        while i<len(query) and not found:
            word = query[i]
            if word in doc_words.get(doc_id):
                docs.append(doc_id)
                found=True
            else:
                i+=1
    return docs


def process(text):
    stoplist = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    st = LancasterStemmer()

    pos_list = pos_tag(word_tokenize(text.lower()))

    word_list = [
        (entry[0], lemmatizer.lemmatize(entry[0], pos[0].lower() if pos[0].lower() in {'a', 'n', 'v'} else 'n'))
        for entry, pos in zip(pos_list, pos_list)
        if entry[0] not in stoplist and entry[0] not in string.punctuation
    ]

    stemmed_lemmatized_wl = [
        (st.stem(entry[0]), entry[1]) for entry in word_list
    ]

    return stemmed_lemmatized_wl


def get_terms(text):
    stoplist = set(stopwords.words('english'))
    terms = {}
    st = LancasterStemmer()
    word_list = [st.stem(word) for word in word_tokenize(text.lower())
                 if not word in stoplist and not word in string.punctuation]
    for word in word_list:
        terms[word] = terms.get(word, 0) + 1
    return terms


def collect_vocabulary(doc_terms, qry_terms):
    all_terms = []
    for doc_id in doc_terms.keys():
        for term in doc_terms.get(doc_id).keys():
            all_terms.append(term)
    for qry_id in qry_terms.keys():
        for term in qry_terms.get(qry_id).keys():
            all_terms.append(term)
    return sorted(set(all_terms))


def vectorize(input_features, vocabulary):
    output = {}
    for item_id in input_features.keys():
        features = input_features.get(item_id)
        output_vector = []
        for word in vocabulary:
            if word in features.keys():
                output_vector.append(int(features.get(word)))
            else:
                output_vector.append(0)
        output[item_id] = output_vector
    return output


def calculate_idfs(vocabulary, doc_features):
    doc_idfs = {}
    for term in vocabulary:
        doc_count = 0
        for doc_id in doc_features.keys():
            terms = doc_features.get(doc_id)
            if term in terms.keys():
                doc_count += 1
        doc_idfs[term] = math.log(float(len(doc_features.keys()))/float(1 + doc_count), 10)
    return doc_idfs


def vectorize_idf(input_terms, input_idfs, vocabulary):
    output = {}
    for item_id in input_terms.keys():
        terms = input_terms.get(item_id)
        output_vector = []
        for term in vocabulary:
            if term in terms.keys():
                output_vector.append(input_idfs.get(term)*float(terms.get(term)))
            else:
                output_vector.append(float(0))
        output[item_id] = output_vector
    return output


def length(vector):
    sq_length = 0
    for index in range(0, len(vector)):
        sq_length += math.pow(vector[index], 2)
    return math.sqrt(sq_length)


def dot_product(vector1, vector2):
    if len(vector1) == len(vector2):
        dot_prod = 0
        for index in range(0, len(vector1)):
            if not vector1[index] == 0 and not vector2[index] == 0:
                dot_prod += vector1[index] * vector2[index]
        return dot_prod
    else:
        return "Unmatching dimensionality"


def calculate_cosine(query, document):
    cosine = dot_product(query, document) / (length(query) * length(document))
    return cosine


def prefilter(doc_terms, query):
    docs = []
    for doc_id in doc_terms.keys():
        found = False
        i = 0
        while i<len(query.keys()) and not found:
            term = list(query.keys())[i]
            if term in doc_terms.get(doc_id).keys():
                docs.append(doc_id)
                found=True
            else:
                i+=1
    return docs


def calculate_precision(model_output, gold_standard):
    true_pos = 0
    for item in model_output:
        if item in gold_standard:
            true_pos += 1
    return float(true_pos)/float(len(model_output))


def calculate_found(model_output, gold_standard):
    found = 0
    for item in model_output:
        if item in gold_standard:
            found = 1
    return float(found)


def load_data():
    documents = read_documents()
    queries = read_queries()
    mappings = read_mappings()
    return documents, queries, mappings

