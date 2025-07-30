from flask import Flask, request, render_template
from pickle import load
from sklearn.feature_extraction.text import CountVectorizer


app = Flask(__name__)


model = load(open('naive-bayes-multinomial.pkl', 'rb'))
class_dict = {"0": "It's a negative comment ):",
              "1": "Its a positiva comment (:"}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template('index.html', prediction=None)
    if request.method == "POST":
        val1 = str(request.form["val1"]).str.strip().str.lower()
        vec_model = CountVectorizer(stop_words = "english")
        predata = vec_model.fit_transform(val1).toarray()
        data = [[predata]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
        return render_template('index.html', prediction=pred_class)
    return None


if __name__ == '__main__':
    # PORT = int(os.environ.get('PORT', 3000))
    app.run(host='0.0.0.0', port=3000, debug=True)