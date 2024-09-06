import gradio as gr
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from pydantic import BaseModel

class MyModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

# Load the model and scaler
model = joblib.load('models/xgb_tunedv2.pkl')
scaler = joblib.load('scalers/scaler.pkl')
columns = joblib.load('models/columns.pkl')

# Columns used during training
columns = ['Duration_to_accept_offer', 'Notice_Period', 'Percent_hike_expected_in_CTC',
           'Percent_difference_CTC', 'Rex_in_Yrs', 'Age', 'DOJ_Extended_Yes', 
           'Offered_Band_E2', 'Offered_Band_E3', 'Joining_Bonus_Yes', 'Candidate_relocate_actual_Yes', 
           'Gender_Male', 'Candidate_Source_Employee Referral', 'LOB_CSMP', 'LOB_EAS', 'LOB_ERS', 
           'LOB_ETS', 'LOB_Healthcare', 'LOB_INFRA', 'LOB_MMS', 'Location_Chennai', 'Location_Cochin', 
           'Location_Gurgaon', 'Location_Hyderabad', 'Location_Kolkata', 'Location_Mumbai', 
           'Location_Noida', 'Location_Pune', 'Region_Name_South', 'Region_Name_West', 
           'Domicile_Name_Bihar', 'Domicile_Name_Chandigarh', 'Domicile_Name_Chhattisgarh', 
           'Domicile_Name_Delhi', 'Domicile_Name_Goa', 'Domicile_Name_Gujarat', 'Domicile_Name_Haryana', 
           'Domicile_Name_Himachal Pradesh', 'Domicile_Name_Jharkhand', 'Domicile_Name_Karnataka', 
           'Domicile_Name_Kerala', 'Domicile_Name_Madhya Pradesh', 'Domicile_Name_Maharashtra', 
           'Domicile_Name_Manipur', 'Domicile_Name_Odisha', 'Domicile_Name_Punjab', 
           'Domicile_Name_Rajasthan', 'Domicile_Name_Tamil Nadu', 'Domicile_Name_Telangana', 
           'Domicile_Name_Tripura', 'Domicile_Name_Uttar Pradesh', 'Domicile_Name_Uttarkhand', 
           'Domicile_Name_West Bengal']

# Preprocessing function
def preprocess_input(input_data):
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode categorical features
    categorical_columns = ['DOJ_Extended', 'Offered_Band', 'Joining_Bonus', 'Candidate_relocate_actual', 
                           'Gender', 'Candidate_Source', 'LOB', 'Location', 'Region_Name', 'Domicile_Name']
    input_df = pd.get_dummies(input_df, columns=categorical_columns, drop_first=True)
    
    # Ensure all columns are present
    for col in columns:
        if col not in input_df.columns:
            input_df[col] = 0
    
    # Scale numeric features
    numeric_features = ['Duration_to_accept_offer', 'Notice_Period', 'Percent_hike_expected_in_CTC', 
                        'Percent_difference_CTC', 'Rex_in_Yrs', 'Age']
    input_df[numeric_features] = scaler.transform(input_df[numeric_features])
    
    return input_df

# Prediction function
def make_prediction(DOJ_Extended, Duration_to_accept_offer, Notice_Period, Offered_Band, Percent_hike_expected_in_CTC,
                    Percent_difference_CTC, Joining_Bonus, Candidate_relocate_actual, Gender, Candidate_Source,
                    LOB, Location, Region_Name, Domicile_Name, Age):

    input_data = {
        'DOJ_Extended': DOJ_Extended,
        'Duration_to_accept_offer': Duration_to_accept_offer,
        'Notice_Period': Notice_Period,
        'Offered_Band': Offered_Band,
        'Percent_hike_expected_in_CTC': Percent_hike_expected_in_CTC,
        'Percent_difference_CTC': Percent_difference_CTC,
        'Joining_Bonus': Joining_Bonus,
        'Candidate_relocate_actual': Candidate_relocate_actual,
        'Gender': Gender,
        'Candidate_Source': Candidate_Source,
        'LOB': LOB,
        'Location': Location,
        'Region_Name': Region_Name,
        'Domicile_Name': Domicile_Name,
        'Age': Age
    }
    
    # Preprocess the input
    processed_input = preprocess_input(input_data)
    
    # Ensure the processed input matches the expected feature order
    processed_input = processed_input[columns]
    
    # Make the prediction
    prediction = model.predict(processed_input)
    
    # Return the prediction result
    return "Yes" if prediction[0] == 1 else "No"

# Gradio UI interface
def inference_page():
    with gr.Blocks() as demo:
        gr.Markdown("## HR Analysis Prediction: Will the Candidate Join?")
        
        DOJ_Extended = gr.Dropdown(choices=["Yes", "No"], label="DOJ Extended")
        Duration_to_accept_offer = gr.Slider(minimum=0, maximum=100, label="Duration to Accept Offer (days)")
        Notice_Period = gr.Slider(minimum=0, maximum=90, label="Notice Period (days)")
        Offered_Band = gr.Dropdown(choices=["E1", "E2", "E3"], label="Offered Band")
        Percent_hike_expected_in_CTC = gr.Slider(minimum=0, maximum=100, label="Percent Hike Expected in CTC")
        Percent_difference_CTC = gr.Slider(minimum=-50, maximum=50, label="Percent Difference in CTC")
        Joining_Bonus = gr.Dropdown(choices=["Yes", "No"], label="Joining Bonus")
        Candidate_relocate_actual = gr.Dropdown(choices=["Yes", "No"], label="Candidate Relocate")
        Gender = gr.Dropdown(choices=["Male", "Female"], label="Gender")
        Candidate_Source = gr.Dropdown(choices=["Direct", "Employee Referral"], label="Candidate Source")
        LOB = gr.Dropdown(choices=["ERS", "CSMP", "BFSI", "EAS", "INFRA", "ETS"], label="Line of Business (LOB)")
        Location = gr.Dropdown(choices=["Noida", "Chennai", "Mumbai", "Hyderabad", "Kolkata", "Gurgaon"], label="Location")
        Region_Name = gr.Dropdown(choices=["North", "South", "West"], label="Region Name")
        Domicile_Name = gr.Dropdown(choices=["Goa", "Himachal Pradesh", "Jharkhand", "Tripura", "Punjab", "Delhi"], label="Domicile Name")
        Age = gr.Slider(minimum=18, maximum=65, label="Age")
        
        predict_button = gr.Button("Predict")
        result = gr.Textbox(label="Prediction Result")

        predict_button.click(make_prediction, 
                             inputs=[DOJ_Extended, Duration_to_accept_offer, Notice_Period, Offered_Band, 
                                     Percent_hike_expected_in_CTC, Percent_difference_CTC, Joining_Bonus, 
                                     Candidate_relocate_actual, Gender, Candidate_Source, LOB, Location, 
                                     Region_Name, Domicile_Name, Age], 
                             outputs=result)

    return demo

# Run the inference page
if __name__ == "__main__":
    inference_page().launch()
