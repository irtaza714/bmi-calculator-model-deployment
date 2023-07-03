from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Gender=str(request.form.get('Gender')),
            Age=int(request.form.get('Age')),
            Height_In_Inches=float(request.form.get('Height_In_Inches')),
            Weight_In_Pounds=float(request.form.get('Weight_In_Pounds'))
        )
        pred_df = data.get_data_as_data_frame()
        print("Before Prediction", pred_df)

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        result = 'The BMI is: {:.2f}'.format (results[0])

        print("After Prediction", results[0])
        return render_template('home.html', results=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
