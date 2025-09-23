from flask import Flask, render_template, request
from generate import generate_rule

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_rule = ""
    if request.method == 'POST':
        log_text = request.form.get('log_text')
        if log_text:
            generated_rule = generate_rule(log_text)
    return render_template('index.html', generated_rule=generated_rule)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
