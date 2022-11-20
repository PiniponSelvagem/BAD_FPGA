
import json
from collections import Counter


def read_json_file(json_file_path:str) -> dict:
    """ Read json file with labels.

                Parameters
                ----------
                json_file_path : str
                    Path of json file.

                Returns:
                -------
                data_dict : list
                    List of labels, each label is a dictionary item with entries 'id', 'start', 'end', 'annotation'
    """
    with open(json_file_path) as json_data:
        data_dict = json.load(json_data)
    return data_dict

def convert_counter_collection_to_counter(counter_collection:[]) -> Counter :
    counter_aggregation = Counter()
    for elmt in counter_collection :
        counter_aggregation += Counter(elmt)
    return counter_aggregation

def ifNone(valueToCheck, defaultValue ) :
    return defaultValue if valueToCheck is None else valueToCheck

def getAccuracy(metrics: Counter) -> float:
    return (metrics.get('TP') + metrics.get('TN')) / (metrics.get('TP') + metrics.get('TN') + metrics.get('FP') + metrics.get('FN'))

def getPrecision(metrics: Counter) -> float:
    return metrics.get('TP') / (metrics.get('TP') + metrics.get('FP'))

def getRecall(metrics: Counter) -> float:
    return metrics.get('TP') / (metrics.get('TP') + metrics.get('FN'))

def getF1(metrics: Counter) -> float:
    return 2 * getPrecision(metrics) * getRecall(metrics) / (getPrecision(metrics) + getRecall(metrics))