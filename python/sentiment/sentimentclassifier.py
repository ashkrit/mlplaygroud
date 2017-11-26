from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB


def load_files(root_folder):
    file_names = ["imdb_labelled.txt", "amazon_cells_labelled.txt", "yelp_labelled.txt"]
    lines = []
    for f in file_names:
        with open(root_folder + f) as text_file:
            lines += text_file.read().split("\n")
    return lines


def transform(lines):
    tab_line = map(lambda line: line.split("\t"), lines)
    valid_lines = filter(lambda line: len(line) == 2 and line[1] <> '', tab_line)
    return valid_lines


def document_labels(lines):
    train_document = map(lambda line: line[0], lines)
    train_label = map(lambda line: int(line[1]), lines)
    return train_document, train_label


def word_index(vocabulary_, key):
    try:
        return vocabulary_[key]
    except KeyError:
        return -1


def build_classifier(train_documents, labels):
    return BernoulliNB().fit(train_documents, labels)


def main():
    print "Start some classification!!!!!!!!!!!!!!!"
    lines = load_files("../../data/sentiment labelled sentences/")
    lines = transform(lines)
    document, labels = document_labels(lines)

    word_vector = CountVectorizer(binary='true')
    document_features = word_vector.fit_transform(document)

    classifier = build_classifier(document_features, labels)

    test = ["this is worst movie", "tlooks like good use of time", "should not go for this"]

    print classifier.predict(word_vector.transform(test))


if __name__ == "__main__":
    main()
