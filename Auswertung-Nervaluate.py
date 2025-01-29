import pandas as pd
import json

file = '' #Evaluation File Path
df = pd.read_parquet(file)

def convert_to_gt_labels(data):
    gt_labels = []  # Zielstruktur

    for array in data:  
        sentence_labels = []
        for dict in array:
            label = dict["label"]  # Das Label (PER, LOC, usw.)
            start = dict["start"]  # Startindex
            end = dict["end"]  # Endindex
            # Strukturiere die Daten um
            sentence_labels.append({"label": label, "start": start, "end": end})
        gt_labels.append(sentence_labels)

    return gt_labels

truth_list = df['answer'].to_numpy()
prediction_list = df['prediction'].to_numpy()

gt_truth = convert_to_gt_labels(truth_list)
gt_prediction = convert_to_gt_labels(prediction_list)

from nervaluate import Evaluator
tags=["Method", "Action", "Reagent", "Sample", "Temperature", "Concentration", "Volume", "Processed", "Labware", "Time", "Equipment", "Waste", "Hint", "Count", "Frequency", "Speed", "Size", "pH", "Current", "Mass", "Pressure", "Power", "Control", "Voltage", "Absorbance", "Measure"]
evaluator = Evaluator(gt_truth, gt_prediction, tags=tags, loader="default")
results = evaluator.evaluate()

output_file = "" #Output Path
with open(output_file, "w") as file:
    json.dump(results, file, indent=4)

print(f"Ergebnisse wurden in {output_file} gespeichert.")
