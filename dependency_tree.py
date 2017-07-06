from nltk import Tree

#doc = nlp("The quick brown fox jumps over the lazy dog.")

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_

def tok_format(tok):
    return "_".join([tok.orth_, tok.tag_])

def find_root(docu):
    for token in docu:
        if token.head is token:
            return token

def to_nltk_tree2(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(tok_format(node), [to_nltk_tree(child) for child in node.children])
    else:
        return tok_format(node)

def to_spacy_desc(node):
	subjects = [w for w in node if w.dep_ == 'nsubj']
	for subject in subjects:
		numbers = [w for w in subject.lefts if w.dep_ == 'nummod']
		if len(numbers) == 1:
			print('subject: {}, action: {}, numbers: {}'.format(subject.text, subject.head.text, numbers[0].text))
#[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]
