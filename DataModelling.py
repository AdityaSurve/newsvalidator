import os
from pyvis.network import Network
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load('en_core_web_md')


def remove_stopwords_and_punctuation(text):
    doc = nlp(text)
    tokens = [
        token.text for token in doc if not token.is_stop and not token.is_punct]
    value = ' '.join(tokens)
    return value.strip()

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
        os.makedirs('output_graphs', exist_ok=True)
        data = pd.read_csv("Relations.csv")
        unique_entities = pd.unique(data['Entity1'])
        for entity in unique_entities:
            net = Network(notebook=True)
            net.force_atlas_2based(spring_length=100)
            entity_data = data[data['Entity1'] == entity]
            relations = entity_data['Relation'].unique()

            for relation in relations:
                related_entities = entity_data[entity_data['Relation']
                                               == relation]['Entity2'].tolist()

                for related_entity in related_entities:
                    net.add_node(entity, color='skyblue', size=50, title=entity,
                                 label=entity, font={"color": "black", "size": 12})
                    net.add_node(related_entity, color='skyblue', size=50, title=related_entity,
                                 label=related_entity, font={"color": "black", "size": 12})
                    net.add_node(relation, color='red', size=50, title=relation,
                                 label=relation, font={"color": "black", "size": 12})
                    net.add_edge(entity, relation, )
                    net.add_edge(relation, related_entity)

            net.show(f"output_graphs/{entity}_relations.html")

    def preprocess(self):
        nlp = spacy.load('en_core_web_md')
        data = pd.read_csv("Relations.csv")

        data['Entity1'] = data['Entity1'].astype(str)
        data['Entity2'] = data['Entity2'].astype(str)
        data['Relation'] = data['Relation'].astype(str)
        data['Entity1'] = data['Entity1'].str.lower()
        data['Entity2'] = data['Entity2'].str.lower()
        data['Relation'] = data['Relation'].str.lower()
        data['Entity1'] = data['Entity1'].apply(
            remove_stopwords_and_punctuation)
        data['Entity2'] = data['Entity2'].apply(
            remove_stopwords_and_punctuation)

        for index, row in data.iterrows():
            if ":" in row['Entity1']:
                data.at[index, 'Entity1'] = row['Entity1'].split(":")[1]
            if ":" in row['Entity2']:
                data.at[index, 'Entity2'] = row['Entity2'].split(":")[1]

        unique_entities1 = pd.unique(data['Entity1'])
        unique_entities2 = pd.unique(data['Entity2'])

        for i in range(len(unique_entities1)):
            for j in range(i+1, len(unique_entities1)):
                entity1 = nlp(unique_entities1[i])
                entity2 = nlp(unique_entities1[j])
                if entity1.has_vector and entity2.has_vector:
                    similarity = entity1.similarity(entity2)
                    if similarity > 0.6:
                        data['Entity1'].replace(
                            unique_entities1[j], unique_entities1[i], inplace=True)

        for i in range(len(unique_entities2)):
            for j in range(i+1, len(unique_entities2)):
                entity1 = nlp(unique_entities2[i])
                entity2 = nlp(unique_entities2[j])
                if entity1.has_vector and entity2.has_vector:
                    similarity = entity1.similarity(entity2)
                    if similarity > 0.8:
                        data['Entity2'].replace(
                            unique_entities2[j], unique_entities2[i], inplace=True)

        data.drop_duplicates(inplace=True)
        data.replace('', np.nan, inplace=True)
        data.dropna(subset=['Entity1', 'Entity2'], inplace=True)
        data.reset_index(drop=True, inplace=True)
        all_entities = pd.concat([data['Entity1'], data['Entity2']])
        all_entities = all_entities.drop_duplicates()
        all_entities = all_entities.reset_index(drop=True)
        all_entities = all_entities.to_frame()
        all_entities.columns = ['Entity']
        all_entities.to_csv("Entities.csv", index=False)
        data.to_csv("Relations.csv", index=False)
