import pandas as pd
import json
import boto3

def add_model_predictions(df, text_column, output_column, sagemaker_client, endpoint_name, instructions):

    predictions = []

    for idx, row in df.iterrows():
        text = row[text_column]
        combined_input = f"{instructions}\n\n{text}"

        try:
            # Request-Payload erstellen
            payload = {"inputs": combined_input, "max_new_tokens": 4000}

            # Anfrage an SageMaker senden
            response = sagemaker_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=json.dumps(payload)
            )

            # Antwort dekodieren und extrahieren (ggf. anpassen)
            response_body = json.loads(response['Body'].read().decode('utf-8'))
            predictions.append(
                response_body)

        except Exception as e:
            predictions.append(None)

    # Predictions-Spalte
    df[output_column] = predictions

    return df


if __name__ == "__main__":
    # Dateipfad anpassen, falls erforderlich
    filepath = '/Users/ender/PycharmProjects/PythonProject1/Files/test.parquet'

    df = pd.read_parquet(filepath)
    # JSONL-Datei laden
    print(f"DataFrame geladen mit {len(df)} Zeilen und den Spalten: {df.columns}")

    # SageMaker-Client erstellen ZUGANGSDATEN ANPASSEN
    aws_access_key_id = ""
    aws_secret_access_key = ""
    aws_session_token = ""
    # SageMaker-Endpunktname
    endpoint_name = ""
    # AWS-Region, in der dein Endpunkt bereitgestellt ist (z. B. eu-central-1, us-west-2)
    region_name = ""

    # SageMaker-Client erzeugen
    sagemaker_client = boto3.client(
        'sagemaker-runtime',
        region_name=region_name,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token
    )

    # 4. prompt definieren
    prompt = """ """
    # 5. Die Funktion aufrufen, um die Vorhersagen zu erhalten
    df = add_model_predictions(df, text_column='question', output_column='predictions',
                               sagemaker_client=sagemaker_client, endpoint_name=endpoint_name,
                               instructions=prompt)

    # Ausgabe des aktualisierten DataFrames
    print(df.head())

    # DataFrame als neue Datei speichern
    output_filepath = ""
    df.to_parquet(output_filepath)
    print(f"Datei m
