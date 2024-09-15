from flask import Flask,render_template,url_for,request,jsonify
import joblib
model = joblib.load("rfmodel.lb")
# model.predict(x_variable)


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/input_form')
def input_form():
    return render_template('input_form.html')

@app.route('/prediction',methods=['GET','POST'])
def prediction():
    if request.method == 'POST':

        region = request.form["region"]
        children = int(request.form["children"])
        gender = int(request.form["gender"])
        age = int(request.form["age"])
        bmi = int(request.form["bmi"])
        smoker = int(request.form["smoker"])

        region_northeast = 0
        region_northwest = 0
        region_southeast = 0
        region_southwest = 0
        if region == "ne":
            region_northeast = 1
        elif region == "nw":
            region_northwest = 1
        elif region == "se":
            region_southeast = 1
        else:
            region_southwest = 1	

        user_data = [[children,smoker,age,bmi,gender,region_northeast,region_northwest,region_southeast,region_southwest]]
        
        pred = model.predict(user_data)[0]
        # return user_data
        return render_template('prediction_page.html',output = str(pred))


if __name__ =="__main__":
    app.run(host="0.0.0.0",port=2525,debug=True) 
