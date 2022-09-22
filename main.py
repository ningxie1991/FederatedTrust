import json
import os

from accountability.assessment import document_completeness

dirname = os.path.dirname(__file__)


def assess():
    print(f'Assessing model')  # Press Ctrl+F8 to toggle the breakpoint.
    with open(os.path.join(dirname, 'assets/factsheet/example/factsheet.json'), 'r') as f, \
        open(os.path.join(dirname, 'assets/factsheet/test/factsheet_missing_properties.json'), 'r') as t:
        factsheet = json.load(f)
        incomplete_factsheet = json.load(t)
        document_completeness(factsheet)
        document_completeness(incomplete_factsheet)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    assess()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
