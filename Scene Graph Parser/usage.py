import sng_parser
import spacy
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")

def generateRelations(sentence):
    # Parse into graph
    graph = sng_parser.parse(sentence)

    # Get and concat embeddings
    relation_vectors = []
    for i in range(len(graph["relations"])):
        # Object
        object = nlp(graph["entities"][graph["relations"][i]["object"]]["head"])
        if len(object) == 1:
            object_embedding = torch.Tensor(object[0].vector).to(device)
        else:
            object = nlp(graph["entities"][graph["relations"][i]["object"]]["lemma_head"])
            object_embedding = torch.Tensor(object[0].vector).to(device)
        # Subject
        subject = nlp(graph["entities"][graph["relations"][i]["subject"]]["head"])
        if len(subject) == 1:
            subject_embedding = torch.Tensor(subject[0].vector).to(device)
        else:
            subject = nlp(graph["entities"][graph["relations"][i]["subject"]]["lemma_head"])
            subject_embedding = torch.Tensor(subject[0].vector).to(device)
        # Relation
        relation = nlp(graph["relations"][i]["relation"])
        if len(relation) == 1:
            relation_embedding = torch.Tensor(relation[0].vector).to(device)
        else:
            relation = nlp(graph["relations"][i]["lemma_relation"])
            relation_embedding = torch.Tensor(object[0].vector).to(device)

        relation_vectors.append(torch.cat([subject_embedding, relation_embedding, object_embedding]))
        print("Relation " + str(i + 1) + ": " + str(relation_vectors[-1]))
        return(relation_vectors)