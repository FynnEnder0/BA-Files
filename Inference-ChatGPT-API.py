import os
import pandas as pd
from openai import AzureOpenAI

# Azure OpenAI Setup
client = AzureOpenAI(
    api_key="",  # API-Key
    api_version="2023-03-15-preview",
    azure_deployment="gpt-4o",# Model-Version
    azure_endpoint=""  # Azure-Endpoint
)


# Funktion zur Klassifikation mit Azure OpenAI
def classify_description(description, prompt_template):

    prompt = f"{prompt_template}\n\n{description}"

    # API-Aufruf
    response = client.chat.completions.create(
        model="chat",
        messages=[{"role": "user", "content": prompt}]
    )

    # Antwort auslesen
    message = response.choices[0].message  # Zugriff auf die Nachricht
    if hasattr(message, "content"):
        return message.content.strip()  # Sicherer Zugriff auf das Feld "content"
    else:
        raise ValueError (''Die Antwort enthaelt kein 'content'-Feld.'')


# Hauptfunktion
if __name__ == "__main__":
    # Eingabedateien definieren
    data_file_path = r""  # Pfad zum Testdatensatz
    output_directory = "Outputs Model"  # Ausgabedateipfad

    # Existiert Ausgabeverzeichnis?
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, "") #Output Dateinamen

    data = pd.read_parquet(data_file_path)

    # Prompt
    promptEval = """ """

    labels = []
    n = 1
    for desc in data["question"]:
        try:
            label = classify_description(desc, promptEval)
            labels.append(label)
            print(f"[{n}] Erfolgreiche Antwort: {label}")
            n+=1
        except Exception as e:
            labels.append("Error")
            print(f"Fehler bei der Verarbeitung: {e}")

    # Ergebnisse in die Daten integrieren
    data["prediction"] = labels

    # Ergebnisse speichern
    try:
        data.to_parquet(output_path)
        print(f"Die Ergebnisse wurden in {output_path} gespeichert.")
    except PermissionError as e:
        print(
            f''PermissionError: Die Datei konnte nicht gespeichert werden. Bitte schliessen Sie die Datei, wenn sie geoeffnet ist, und versuchen Sie es erneut'')
