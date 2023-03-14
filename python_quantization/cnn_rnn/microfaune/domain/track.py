from functools import reduce
from collections import Counter
from utils import misc_utils as util
import operator
import numpy as np

# PREDICTION_SAMPLE_WIDTH_IN_MS = 20

class Track:

    # "tracks": [
    #     {
    #         "id": 47,
    #         "name": "SWIFT_20000101_022052.wav",
    #         "file": "/media/SWIFT_20000101_022052.wav",
    #         "format": "wav",
    #         "project_id": 1,
    #         "duration": 60.0,
    #         "prediction": "",
    #         "annotation_set": [
    #             {
    #                 "id": 1,
    #                 "track_id": 47,
    #                 "value":"[
    #                   {\"id\":\"wavesurfer_4kgitqcktig\",\"start\":4.935061842665718,\"end\":10.509955195406347,\"annotation\":\"\"},
    #                   {\"id\":\"wavesurfer_afb13jpasm8\",\"start\":17.55982033205593,\"end\":22.95971703246838,\"annotation\":\"\"},
    #                   {\"id\":\"wavesurfer_jdu8bguik4\",\"start\":26.334652470226157,\"end\":30.184578821446145,\"annotation\":\"\"}
    #                 ]",
    #           "user_id": 1,
    #           "reviewed": false,
    #           "reviewed_by_id": null,
    #           "date_time": "2020-10-17T19:12:13.800Z",
    #           "username": "admin",
    #           "reviewer": null
    #           }
    #         ] --> Fin annotation_set
    #     }, --> Fin du TrackElmt
    # ] --> Fin du Tracks


    ############################
    # Compute positive indexes #
    ############################
    def map_annotation_set_to_prediction_ndxes(self, track_elmt:dict) -> [(int,int)] :
        '''
        :param track_elmt: An element of 'tracks'
        :return: An array of (start,end) excerpt corresponding to the indexes in Prediction structure of Tack[]
        '''
        prediction_ndxes_of_annotation_set_elmt = []
        prediction_ndxes_of_annotation_set_elmt +=  map(lambda annotation_set_elmt :
            self._map_annotation_set_elmt_to_prediction_ndxes(annotation_set_elmt.get("value"), track_elmt.get("duration"), len(track_elmt.get("prediction")) ) ,
                                                              track_elmt.get("annotation_set"))
        prediction_ndxes_of_annotation_set_elmt = reduce(operator.concat, prediction_ndxes_of_annotation_set_elmt, [])
        return prediction_ndxes_of_annotation_set_elmt

    def _map_annotation_set_elmt_to_prediction_ndxes(self, value_list:[dict], track_duration:float, track_predictions_count:int ) -> [(int,int)]:
        '''
        :param value_list: track.annotation_set.value[]
        :param track_duration: tracks.duration
        :param track_predictions_count: tracks.prediction.length
        :return: An array of (start,end) excerpt corresponding to the indexes in Prediction structure of a particular Tack
        '''
        prediction_ndxes_of_value_elmt = []
        prediction_ndxes_of_value_elmt += map(lambda value : self._convert_from_annonation_elmt_time_to_prediction_order(value, track_duration, track_predictions_count),
                                              value_list)
        return prediction_ndxes_of_value_elmt


    def _convert_from_annonation_elmt_time_to_prediction_order(self, value:dict, track_duration:float, track_predictions_count:int ) -> (int,int):
        '''
                Formula of a tuple (start,end) excerpt corresponding to the indexes in Prediction structure of a particular track.value
                track_duration(60 sec) ---represente-par--> track.prediction.length (2814)
                start                  ------------------->  ?
        :param value: track.value
        :param track_duration: tracks.duration
        :param track_predictions_count: tracks.prediction.length
        :return: An tuple (start,end) excerpt corresponding to the indexes in Prediction structure of a particular track.value
        '''
        return (  int(value.get("start") * track_predictions_count // track_duration) ,
                  int(value.get("end") * track_predictions_count // track_duration)  )


    ####################
    # Compute Metrics  #
    ####################
    def compute_metrics_of_prediction_against_annotation(self, track_elmt:dict) -> Counter :
        print(f'**Track** id:{track_elmt.get("id")} / name:{track_elmt.get("name")} / file:{track_elmt.get("file")} / duration:{track_elmt.get("duration")}')
        positive_tuples_ndexes = self.map_annotation_set_to_prediction_ndxes(track_elmt)
        return self.compute_track_elmt_metrics(positive_tuples_ndexes, track_elmt.get("prediction"))

    def compute_track_elmt_metrics(self, positive_tuples_ndexes : [(int, int)], predictions:[]) -> Counter :
        '''
        :param positive_tuples_ndexes: list of positive annotated tuple indexes
        :param predictions: list of predictions made by the model
        :return: Counter representing the confusion matrix metrics TP / TN / FP / FN
        '''
        positive_tuples_ndexes.sort(key= lambda tuple: tuple[0])
        # 1 - Compute TP and FN
        counter_tpfn = self._do_compute_track_elmt_metrics(positive_tuples_ndexes , predictions, 'tp_and_fn')
        # 2 - Compute 'negative_tuples_ndexes' used to compute TN and FFP
        negative_tuples_ndexes = self._compute_negative_tuples_ndexes(positive_tuples_ndexes, predictions)
        # 3 - Compute FP and TN
        counter_fptn = self._do_compute_track_elmt_metrics(negative_tuples_ndexes , predictions, 'fp_and_tn')
        # 4 - Return the Confusion Matrix element
        metrics =  Counter({'TP': util.ifNone(counter_tpfn.get(1.0), 0.0), 'FN': util.ifNone(counter_tpfn.get(0.0), 0.0),
                            'FP': util.ifNone(counter_fptn.get(1.0), 0.0), 'TN': util.ifNone(counter_fptn.get(0.0), 0.0) })
        print(f'\t{metrics}')
        return metrics

    def _do_compute_track_elmt_metrics(self, positive_or_negative_tuples_ndexes : [(int, int)], predictions:[], n_p_desc : str) -> Counter :
        p_and_n = list(map(lambda positive_or_negative_tuple_ndexes: self._compute_metrics_according_to_ground_truth(positive_or_negative_tuple_ndexes, predictions),
                             positive_or_negative_tuples_ndexes))
        counter_positive_negative = util.convert_counter_collection_to_counter(p_and_n)
        # print(f'{n_p_desc} : {p_and_n}\ncounter_{n_p_desc} : {counter_positive_negative}\n')
        return counter_positive_negative


    def _compute_metrics_according_to_ground_truth(self, tuple_ndexes:(int, int), predictions:[]) -> Counter : #() :
        '''
        :param tuple_ndexes: TRUE positive (or) TRUE negative tuple index, according to the ground of truth 'tha annotation'
        :param predictions: predictions according to the RNN model prediction
        :return: In case of tuple_ndexes represents TRUE positive => Counter dictionary { 0:x , 1:y} holding the number of FN (the 0) and TP (the 1)
                 In case of tuple_ndexes represents TRUE negative => Counter dictionary { 0:x , 1:y} holding the number of TN (the 0) and FP (the 1)
        '''
        return Counter(np.floor(np.dot(np.array(predictions[tuple_ndexes[0]:tuple_ndexes[1] + 1:], np.float32), 2)) )


    def _compute_negative_tuples_ndexes(self, positive_tuples_ndexes : [(int, int)], predictions:[]) -> [(int, int)] :
        negative_tuples_ndexes = []
        # 1 - Set the 1st elmt of 'negative_tuples_ndexes'. It is likely before the 1st elmt of 'positive_tuples_ndexes' BUT not sure
        if positive_tuples_ndexes[0][0] != 0 :
            negative_tuples_ndexes.append( (0, positive_tuples_ndexes[0][0]-1) )
        # 2 - Initialize the remaining element of 'negative_tuples_ndexes' BUT beware how initalizing the last elmt
        for i in range(0, len(positive_tuples_ndexes)) :
            negative_tuples_ndexes.append( (positive_tuples_ndexes[i][1]+1,
                                            positive_tuples_ndexes[i+1][0]-1  if i < len(positive_tuples_ndexes) - 1
                                            else len(predictions) - 1 ) )
        # 3 - Check that the last index did't exceed the prediction size
        if negative_tuples_ndexes[-1][0] > len(predictions) - 1 :
            negative_tuples_ndexes.pop()
        return negative_tuples_ndexes