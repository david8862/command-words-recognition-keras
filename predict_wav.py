#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import argparse
import numpy as np

from tensorflow.keras.models import load_model

from train import get_classes
from data import get_mfcc_feature
from params import inject_params


def predict_wav(model_path, wav_path, class_names, top_k):
    model = load_model(model_path, compile=False)

    feature_data = get_mfcc_feature(wav_path)

    feature_data = np.expand_dims(feature_data, axis=0)
    predictions = model.predict(feature_data)[0]

    # get top_k result
    sorted_index = np.argsort(predictions)[::-1]
    for i in range(top_k):
        index = sorted_index[i]
        human_string = class_names[index]
        score = predictions[index] * 100
        print('%s (%3.2f%%)' % (human_string, score))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='model file to predict', type=str, required=True)
    parser.add_argument('--wav_path', help='WAV sound file path.', type=str, required=True)
    parser.add_argument('--classes_path', help='path to class name definitions', type=str, required=True)
    parser.add_argument('--params_path', help='path to params json file', type=str, required=False, default=None)
    parser.add_argument('--top_k', help='top k prediction to print, default=%(default)s.', type=int, required=False, default=1)

    args = parser.parse_args()

    class_names = get_classes(args.classes_path)

    # load & update audio params
    if args.params_path:
        inject_params(args.params_path)

    predict_wav(args.model_path, args.wav_path, class_names, args.top_k)


if __name__ == '__main__':
    main()
