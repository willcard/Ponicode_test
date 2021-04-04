from flask import Flask, render_template, request, jsonify
from preprocess import *
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET','POST'])
def main():
    return render_template('index.html')



@app.route('/call', methods=['POST'])
def call():
    search_str = request.values.get('user_function')

    try:
        inputs = create_inputs(search_str)
    except:
        return jsonify({'response':['<b class="text-danger">Could not create inputs from this function</b>']})

    outputs = {p:model.predict(v)[0] for p,v in inputs.items()}
    print(outputs)
    return jsonify({'response':[f"<b>{param}</b> : {type} " for param,type in outputs.items()]})



if __name__ == "__main__":
    app.run(host="localhost", debug=True)
