import os
import re
import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")


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
