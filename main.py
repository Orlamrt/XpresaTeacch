from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import json

app = Flask(__name__)
CORS(app)  # Aplicar CORS a la aplicación Flask

# Cargar el archivo JSON
with open('conectores.json', 'r') as file:
    data = json.load(file)

# Extraer la lista de conectores
conectores = data['conectores']

def determinar_conector(sentence, conectores):
    tokenizer = GPT2Tokenizer.from_pretrained("datificate/gpt2-small-spanish")
    model = GPT2LMHeadModel.from_pretrained("datificate/gpt2-small-spanish")

    confianza_conectores = {}

    for conector in conectores:
        input_text = "¿Cuál es el mejor conector para: " + sentence + " " + conector + " ?"
        input_ids = tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad():
            outputs = model(input_ids)
            predictions = outputs.logits[0, -1, :]

        k = 5
        top_k_probabilities = torch.topk(predictions, k).values
        confianza = torch.sum(top_k_probabilities).item()

        confianza_conectores[conector] = confianza

    mejor_conector = max(confianza_conectores, key=confianza_conectores.get)

    return mejor_conector

def insertar_conector(sentence, conectores):
    conector = determinar_conector(sentence, conectores)
    words = sentence.split()
    last_word_index = len(words) - 1
    words.insert(last_word_index, conector)
    return ' '.join(words)

@app.route('/insertar_conector', methods=['POST'])
def insertar_conector_api():
    data = request.json
    sentence = data['sentence']
    frase_con_conector = insertar_conector(sentence, conectores)
    return jsonify({"frase_con_conector": frase_con_conector})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

