from flask import Flask, render_template, request, jsonify
import data_method as dm
import json
from keras.models import load_model

app = Flask(__name__)
model = load_model('Clindata_LSTM_model1.h5')

@app.route('/')
def main():
    return render_template('index.html')


@app.route('/query', methods=['GET'])
def query():
    age = request.args['age']
    age_group = request.args['age-group']
    gender = request.args['gender']
    weight = request.args['weight']
    seq_list = request.args.getlist('sequence')
    rxcui_list = request.args.getlist('rxcui')
    reported_roles_list = request.args.getlist('reported-role')
    dechal_list = request.args.getlist('dechal')
    rechal_list = request.args.getlist('rechal')
    indications_list = request.args.getlist('indications')
    json_file = {'age': age, 'age_group': age_group, 'gender': gender, 'weight': weight,
            'sequences': seq_list, 'rxcuis': rxcui_list, 'reported_roles': reported_roles_list,
            'dechals': dechal_list, 'rechals': rechal_list, 'indications': indications_list}
    response = dm.run_model(json_file, model)
    json_response = json.loads(response)
    data = []
    index = []
    if 'data' in json_response:
        data = json_response['data']
    if 'index' in json_response:
        index = json_response['index']
    list_data = zip(index, data)
    return render_template('result.html', data=list_data)

@app.route('/getRxcuis', methods=['get'])
def get_rxcui():
    rxcuis = dm.get_rxcui()
    return jsonify(rxcuis)

@app.route('/getIndications', methods=['get'])
def get_indications():
    indications = dm.get_indication()
    return jsonify(indications)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
