import MySQLdb
from flask import Flask, render_template, request, jsonify


app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/query',  methods=['GET'])
def query_ddi():
    context = []
    _drug1 = request.args['drug1']
    _drug2 = request.args['drug2']
    _drug1_arr = _drug1.split(',')
    if _drug1 and _drug2:
        db = MySQLdb.connect("localhost","root","root","clindata")
        cursor = db.cursor()

        query_string = "SELECT DISTINCT name_2, severity, interaction FROM new_ddi WHERE name_1 ='" + _drug2 + "' and ("
        for q in _drug1_arr:
             query_string = query_string + " name_2='" + q + "' or";
        query_string = query_string[:-3] + ");"
        print(query_string)
        cursor.execute(query_string)
        data = cursor.fetchall()
        msg = {'new_drug': _drug2}
        if len(data) == 0:
            msg['error'] = 'No result found!'
        db.close()
    return render_template('index.html', data=data, msg=msg)

@app.route('/getDrugs', methods=['GET'])
def get_drugs():
    db = MySQLdb.connect("localhost", "root", "root", "clindata")
    cursor = db.cursor()
    query_string = "SELECT DISTINCT name_1 FROM new_ddi";
    cursor.execute(query_string)
    data = cursor.fetchall()
    db.close()
    list_data = [element for d in data for element in d]
    return jsonify(list_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
