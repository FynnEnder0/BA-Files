    import google.generativeai as genai
import os
import pandas as pd
from google.api_core import retry
from google.generativeai.types import RequestOptions

genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")


def classify_description(description, prompt_template):

    prompt = f"{prompt_template}\n\n{description}"

    # API-Aufruf
    response = model.generate_content(prompt, request_options=RequestOptions(
        retry=retry.Retry(initial=1, multiplier=1, maximum=60, timeout=10)))
    print(response.text)

    return response.text
    # Antwort auslesen

if __name__ == "__main__":
    # Eingabedateien definieren
    data_file_path = r""  # Testdatensatz-Pfad
    output_directory = ""  # Ausgabedatei-Pfad

    # Existiert Ausgabepfad?, sonst erstellen
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    output_path = os.path.join(output_directory, "") #Ausgabedateiname

    # Excel-Dateien einlesen
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
