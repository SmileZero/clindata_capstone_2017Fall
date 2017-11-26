import MySQLdb, itertools
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/query',  methods=['GET'])
def query_ddi():
    context = []
    _rxcui1 = request.args['rxcui1']
    _rxcui2 = request.args['rxcui2']
    _rxcui1_arr = _rxcui1.split(',')
    if _rxcui1 and _rxcui2:
        db = MySQLdb.connect("localhost","root","root","clindata")
        cursor = db.cursor()
        query_string = "SELECT rxcui_2, name_2, severity, interaction FROM ddi WHERE rxcui_1 =" + _rxcui2 + " and ("
        for q in _rxcui1_arr:
             query_string = query_string + " rxcui_2=" + q + " or";
        query_string = query_string[:-3] + ");"
        cursor.execute(query_string)
        data = cursor.fetchall()
        msg = {'new_drug': _rxcui2}
        if len(data) == 0:
            msg['error'] = 'No result found!'
        db.close()
    return render_template('index.html', data=data, msg=msg)

@app.route('/getRxcuis', methods=['GET'])
def get_rxcuis():
    db = MySQLdb.connect("localhost", "root", "root", "clindata")
    cursor = db.cursor()
    query_string = "SELECT DISTINCT rxcui_1 FROM ddi";
    cursor.execute(query_string)
    data = cursor.fetchall()
    db.close()
    list_data = [element for d in data for element in d]
    return jsonify(list_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
