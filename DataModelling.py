from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
import pandas as pd
import numpy as np
import spacy
import matplotlib
matplotlib.use('Agg')

class DataModel:
    def __init__(self):
        if os.path.exists("Entities.csv") and os.path.getsize("Entities.csv") > 0:
            self.entities = pd.read_csv("Entities.csv")
        else:
            self.entities = pd.DataFrame(columns=["Entity"])

        if os.path.exists("Relations.csv") and os.path.getsize("Relations.csv") > 0:
            self.relations = pd.read_csv("Relations.csv")
        else:
            self.relations = pd.DataFrame(
                columns=["Relation", "Entity1", "Entity2"])

    def get_entity(self, entity_name):
        return self.entities[self.entities["Entity"] == entity_name]

    def add_entity(self, entity_name):
        entity_name = entity_name.strip('\'"')
        if len(self.entities[self.entities["Entity"] == entity_name]) == 0:
            new_entity = pd.DataFrame({"Entity": [entity_name]})
            self.entities = pd.concat(
                [self.entities, new_entity], ignore_index=True)

    def get_relations(self, relation, entities):
        entity1 = entities[0]
        entity2 = entities[1]
        return self.relations[(self.relations["Relation"] == relation) & (self.relations["Entity1"] == entity1) & (self.relations["Entity2"] == entity2)]

    def add_relation(self, relation, entities):
        entity1 = entities[0]
        entity2 = entities[1]
        if len(self.relations[(self.relations["Relation"] == relation) & (self.relations["Entity1"] == entity1) & (self.relations["Entity2"] == entity2)]) == 0:
            new_relation = pd.DataFrame(
                {"Relation": [relation], "Entity1": [entity1], "Entity2": [entity2]})
            self.relations = pd.concat(
                [self.relations, new_relation], ignore_index=True)

    def get_all_entities(self):
        return self.entities["Entity"].values

    def get_all_relations_by_entity(self, entity_name):
        relations = []
        if len(self.entities[self.entities["Entity"] == entity_name]) != 0:
            for r in self.relations.iterrows():
                r = r[1]
                if entity_name == r['Entity1'] or entity_name == r['Entity2']:
                    relations.append(r)
        return relations

    def save_to_csv(self):
        self.entities.to_csv("Entities.csv", index=False)
        self.relations.to_csv("Relations.csv", index=False)

    def print(self):
        relation_df = self.relations.to_dict(orient='records')
        G = nx.DiGraph()
        for item in relation_df:
            G.add_node(item["Entity1"])
            G.add_node(item["Entity2"])
            G.add_edge(item["Entity1"], item["Entity2"],
                       relation=item["Relation"])
        pos = nx.shell_layout(G)
        nx.draw(G, pos, node_color='skyblue',
                node_size=6000, edge_cmap=plt.cm.Blues)
        for node, (x, y) in pos.items():
            words = node.split(' ')
            lines = [' '.join(words[i:i+2]) for i in range(0, len(words), 2)]
            plt.annotate(
                '\n'.join(lines),
                xy=(x, y), textcoords='offset points',
                horizontalalignment='center', verticalalignment='center',
                fontsize=6, weight='bold'
            )
        edge_labels = nx.get_edge_attributes(G, 'relation')
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=6)
        plt.figure(figsize=(10, 10))
        nx.draw(G, pos, node_color='skyblue',
                node_size=1200, with_labels=True)
        plt.savefig('graph.png')

    def preprocess_relations(self):
        nlp = spacy.load('en_core_web_md')
        data = pd.read_csv("Relations.csv")
        unique_entities1 = pd.unique(data['Entity1'])
        unique_entities2 = pd.unique(data['Entity2'])
        for i in range(len(unique_entities1)):
            for j in range(i+1, len(unique_entities1)):
                entity1 = nlp(unique_entities1[i])
                entity2 = nlp(unique_entities1[j])
                similarity = entity1.similarity(entity2)
                if similarity > 0.6:
                    data['Entity1'].replace(
                        unique_entities1[j], unique_entities1[i], inplace=True)

        for i in range(len(unique_entities2)):
            for j in range(i+1, len(unique_entities2)):
                entity1 = nlp(unique_entities2[i])
                entity2 = nlp(unique_entities2[j])
                similarity = entity1.similarity(entity2)
                if similarity > 0.8:
                    data['Entity2'].replace(
                        unique_entities2[j], unique_entities2[i], inplace=True)

        data.drop_duplicates(inplace=True)
        data.reset_index(drop=True, inplace=True)
        data.to_csv("Relations.csv", index=False)
