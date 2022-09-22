import json
import jsonschema
import os

from jsonschema import ValidationError
from flatten_json import flatten

dirname = os.path.dirname(__file__)


def document_completeness(factsheet):
    print("Assessing document completeness")
    completeness = 0
    with open(os.path.join(dirname, '../assets/factsheet/example/factsheet.json'), 'r') as f, \
            open(os.path.join(dirname, '../assets/factsheet/schema/factsheet_schema.json'), 'r') as s:
        example = json.load(f)
        schema = json.load(s)
        try:
            jsonschema.validate(factsheet, schema)
            completeness = 1
        except ValidationError as e:
            print(e.message)
            completeness = len(flatten(factsheet)) / len(flatten(example))
    print(f"completeness score: {completeness}")
    return completeness
