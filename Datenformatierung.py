import pandas as pd
from transformers import AutoTokenizer
import dotenv
from huggingface_hub import login
import os
import utils
import random
from sklearn.model_selection import train_test_split
from textwrap import dedent
from datasets import Dataset, load_dataset
import ast  # Zum Parsen von Strings in Arrays
import numpy as np

file_path = '' # Datensatz Pfad
dfFormatted = pd.read_parquet(file_path)

dotenv.load_dotenv()
login(token=os.getenv("HF_LOGING_TOKEN"))

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SYSTEM = "You are an expert for annotating named entities from laboratory protocols. Find all relevant entities in the protocol and annoatate them with the correct label from start index to end index. The index starts a 0 at the beginning of the text"
PAD_TOKEN = "<|pad|>"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
tokenizer.padding_side = "right"

def format_example(row: dict):
    prompt = dedent(f"{row["question"]}")
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": row["answer"]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False)

def convert_to_gt_labels(data):
    gt_labels = []  # Zielstruktur

    for array in data:
        sentence_labels = []
        for second_array in array:
            for dict in second_array:
                label = dict["label"]  # Das Label ('Action', 'Labware', usw.)
                start = dict["start"]  # Startindex
                end = dict["end"]  # Endindex
                # Strukturiere die Daten um
                sentence_labels.append({"label": label, "start": start, "end": end})
            gt_labels.append(sentence_labels)

    return gt_labels

# Parquet-Datei lesen
df5 = pd.read_parquet(file_path)
column_list = df5['answer'].to_numpy()

gt_labels = convert_to_gt_labels(column_list)

dfConcat['answer'] = gt_labels

def splitAndConvert(df):
    # Seed generieren und setzen
    seed = 9484
    print(f"Verwendeter Seed: {seed}")

    training_df, test_df = train_test_split(df, test_size=0.20, random_state=seed, shuffle=True)


    train_text = training_df[['text']]
    test_text = test_df[['text']]

    # DataFrames in JSONL-Dateien speichern
    train_text.to_json('train_data.jsonl', orient="records", lines=True)
    test_text.to_json('validation_data.jsonl', orient="records", lines=True)

splitAndConvert(dfFormatted)


dfConcat['text'] = dfConcat.apply(format_example, axis = 1)

dfConcat.to_parquet('dfConcatFinalHalf.parquet')

splitAndConvert(dfConcat)
