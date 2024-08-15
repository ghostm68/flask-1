from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data['prompt']

    # Tokenize the prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    # Generate text
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)

    # Decode the generated output
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text as JSON
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)  # Run on all interfaces, port 8080

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# ... your existing code ...

@app.route('/generate', methods=['POST'])
def generate_text():
    # ... your code to process the prompt and generate text ...
    return jsonify({'generated_text': generated_text})

if __name__ == '__name__':
    app.run(host='0.0.0.0')
