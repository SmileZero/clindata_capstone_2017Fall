import MySQLdb
from flask import Flask, render_template, request


app = Flask(__name__)


@app.route('/')
def main():
    return render_template('index.html')

@app.route('/query',  methods=['GET'])
def query_ddi():
    context = []
    _rxcui1 = request.args['rxcui1']
    _rxcui2 = request.args['rxcui2']
    if _rxcui1 and _rxcui2:
        db = MySQLdb.connect("localhost","root","root","clindata")
        cursor = db.cursor()
        query_string = "SELECT severity, reaction FROM ddi WHERE rxcui_1 =" + _rxcui1 + " and rxcui_2=" + _rxcui2;
        cursor.execute(query_string)
        data = cursor.fetchall()
        db.close()
        for x in data:
            context.append({'severity': x[0], 'interaction': x[1]})
        msg = {'rxcui1': _rxcui1, 'rxcui2': _rxcui2}
        if len(data) == 0:
            msg['error'] = 'No result found!'
    return render_template('index.html', context=context, msg=msg)

if __name__ == '__main__':
    app.run(host='0.0.0.0')
