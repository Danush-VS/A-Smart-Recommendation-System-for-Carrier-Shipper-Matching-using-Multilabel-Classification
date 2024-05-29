from flask import *
import os,sys
app = Flask(__name__)

user_id = 'admin'
user_pwd = 'admin123'

@app.route('/')
def log():
    return render_template('login.html')
@app.route('/submit_log',methods=['GET','POST'])
def log_sub():
    if request.method=='POST':
        uid = request.form['uid']
        upwd = request.form['pwd']
        error = 'Invalid Credentials'

        if uid == user_id and upwd == user_pwd:
            return render_template('index.html')
        else:
            return render_template('login.html',error=error)

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/eda')
def eda():
    return render_template('eda.html')

@app.route('/detector')
def detector():
    return render_template('detector_new.html')

@app.route('/model_parameter')
def model_parameter():
    return render_template('model_parameter.html')


@app.route('/submit_detector', methods=['POST'])
def choose_file():
    if request.method == 'POST':
        
        weight = float(request.form['weight'])
        price = float(request.form['price'])
        length = float(request.form['length'])
        width = float(request.form['width'])
        height = float(request.form['height'])
        product_category = float(request.form['product_category'])
        shipping_distance = float(request.form['shipping_distance'])
        fragility = float(request.form['fragility'])
        urgency = float(request.form['urgency'])

        import joblib
        scaler = joblib.load(r'0_scaler.pkl')
        svm_model = joblib.load(r'1_svm.pkl')
        rf_model = joblib.load(r'2_rf.pkl')
        dt_model = joblib.load(r'3_dt.pkl')
        hybrid_model = joblib.load(r'4_hybrid_model.pkl')


        weight = float(weight)
        price = int(price)
        length = float(length)
        width = float(width)
        height = float(height)
        product_category = float(product_category)
        shipping_distance = int(shipping_distance)
        fragility = int(fragility)
        urgency = int(urgency)

        
        new_user_input = [[weight,price,length,
        width,height,product_category,
        shipping_distance,fragility,urgency]]

        new_user_input = scaler.transform(new_user_input)

        
        mode_of_transport_num_to_text = {0: 'Air', 1: 'Sea', 2: 'Land'}

        svm_model_output = svm_model.predict(new_user_input)[0]
        svm_model_recommendation = mode_of_transport_num_to_text[svm_model_output]

        rf_model_output = rf_model.predict(new_user_input)[0]
        rf_model_recommendation = mode_of_transport_num_to_text[rf_model_output]

        dt_model_output = dt_model.predict(new_user_input)[0]
        dt_model_recommendation = mode_of_transport_num_to_text[dt_model_output]

        hybrid_model_output = hybrid_model.predict(new_user_input)[0]
        hybrid_model_recommendation = mode_of_transport_num_to_text[hybrid_model_output]

        
        print(f"svm recommendation : {svm_model_recommendation}")
        print(f"decision tree recommendation : {dt_model_recommendation}")
        print(f"random forest recommendation : {rf_model_recommendation}")
        print(f"hybrid model recommendation : {hybrid_model_recommendation}")
        
        return render_template("detector_new.html",svm_model_recommendation=weight,dt_model_recommendation=dt_model_recommendation,rf_model_recommendation=rf_model_recommendation,hybrid_model_recommendation=hybrid_model_recommendation)
        

if __name__=='__main__':
    app.run(debug=True)




    