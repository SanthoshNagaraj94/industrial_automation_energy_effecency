from flask import Flask,render_template,request,redirect,url_for
import pickle


model=pickle.load(open('model.pkl','rb'))
scale=pickle.load(open('scale.pkl','rb'))


# app = Flask(__name__) # to make the app run without any
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():

    return render_template("index.html")

@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method=="POST":
        req=request.form

        X1 = float(req["X1"])
        X2 = float(req["X2"])
        X3 = float(req["X3"])
        X4 = float(req["X4"])
        X5 = float(req["X5"])
        X6 = float(req["X6"])
        X7 = float(req["X7"])
        X8 = float(req["X8"])
        x=[[X1,X2,X3,X4,X5,X6,X7,X8]]

        X_scale=scale.transform(x)
        predict=model.predict(X_scale)



        return render_template("result.html", value=predict)




if __name__=="__main__":

    app.run(debug=True)










