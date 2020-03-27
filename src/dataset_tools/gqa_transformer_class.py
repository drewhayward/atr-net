import json
import os
from pathlib import Path

from src.dataset_tools.dataset_transformer_class import DatasetTransformer

class GQATransformer(DatasetTransformer):
    """Extends DatasetTransformer for GQA annotation"""

    def __init__(self, config):
        """Initialize transformer"""
        super().__init__(config)

    def transform(self):
        """Run the transformation pipeline."""
        jsons = [
            self._preddet_json,
            self._predcls_json,
            self._predicate_json,
            self._object_json,
            self._word2vec_json,
            self._preddet_probability_json,
            self._predcls_probability_json
        ]
        if not all(os.path.exists(anno) for anno in jsons):
            annos = self.create_relationship_json()
            predicates, objects = self.save_predicates_objects(annos)
            if not os.path.exists(self._word2vec_json):
                self.save_word2vec_vectors(predicates, objects)
            annos = self.update_labels(annos, predicates, objects)
            if not os.path.exists(self._predcls_json):
                self.create_pred_cls_json(annos, predicates)
            if not os.path.exists(self._preddet_probability_json):
                with open(self._preddet_json) as fid:
                    annos = json.load(fid)
                self.compute_relationship_probabilities(
                    annos, predicates, objects, with_bg=False)
            if not os.path.exists(self._predcls_probability_json):
                with open(self._predcls_json) as fid:
                    annos = json.load(fid)
                self.compute_relationship_probabilities(
                    annos, predicates, objects, with_bg=True)

    def create_relationship_json(self):
        self._load_dataset()
        return self._set_annos()

    def _load_dataset(self):
        """Load the scene graphs"""
        with open(self._orig_annos_path + '/sceneGraphs/train_sceneGraphs.json') as file:
            self._train_scene_graphs = json.load(file)

        with open(self._orig_annos_path + '/sceneGraphs/val_sceneGraphs.json') as file:
            self._val_scene_graphs = json.load(file)


    def _gqa_to_annos(self, gqa_object, split):
        """
        Converts a GGA scene graph file into a annotation object

        Inputs:
            - gqa_object: object directly read from the json file
            - split: int, which split these annotations are in 0/1/2
                for train/val/test respectively
        """
        annos = []
        for scene_id, scene in gqa_object.items():
            
            obj_keys = list(scene['objects'].keys())

            obj_names = []
            rel_names = []
            subject_ids = []
            object_ids = []
            boxes = []

            for i, obj_key in enumerate(obj_keys):
                obj = scene['objects'][obj_key]
                obj_names.append(obj['name'])
                boxes.append(self._convert_box(obj['x'], obj['y'], obj['w'], obj['h']))

                for relation in obj['relations']:
                    rel_names.append(relation['name'])
                    subject_ids.append(i)
                    object_ids.append(obj_keys.index(obj_key))

            annos.append({
                'filename': scene_id + '.jpg',
                'split_id': split,
                'height': scene['height'],
                'width': scene['width'],
                'im_scale': self._compute_im_scale(scene_id + '.jpg'),
                'objects': {
                    'names': obj_names,
                    'boxes': boxes
                },
                'relations': {
                    'names':rel_names,
                    'subj_ids': subject_ids,
                    'obj_ids': object_ids
                }
            })
        return annos

    def _set_annos(self):

        annos = self._gqa_to_annos(self._train_scene_graphs, 0)
        annos += self._gqa_to_annos(self._train_scene_graphs, 1)
        return annos

    @staticmethod
    def _convert_box(x, y, width, height):
        """
        Converts GQA bbox format to y_min, y_max, x_min, x_max

        Inputs:
            - x: int, x value of the top left corner
            - y: int, y value of the top left corner
            - height: int, height of the box
            - width: int, width of the box

        Returns:
            - decoded box: list, [y_min, y_max, x_min, x_max)
        """
        return [y, y + height, x, x + width]