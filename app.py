from flask import Flask, render_template, request
from transformers import pipeline
from flask import escape  # Import the escape function

app = Flask(__name__)

# Load the GPT-Neo model
generator = pipeline('text-generation', model='EleutherAI/gpt-neo-125M')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        output = generator(text, max_length=100, num_return_sequences=1)[0]['generated_text']
        # Escape the output before rendering
        return render_template('index.html', output=escape(output))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
