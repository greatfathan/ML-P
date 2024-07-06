import pandas as pd
import pickle
import streamlit as st

import streamlit.components.v1 as components
import shap

def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('artifacts/explainer.pkl', 'rb') as f:
    explainer = pickle.load(f)
    
with st.sidebar.form(key='house_price'):
    
    form_input = {
        'Bedrooms': st.number_input('Bedrooms', value=3),
        'Bathrooms': st.number_input('Bathrooms', value=2),
        'Garage': st.number_input('Garage', 2),
        'Build Year': st.number_input('Build Year', 2000),
        'Floor Area': st.number_input('Floor Area', 200),
    }
    
    name = st.text_input('Address', placeholder='Wall Street')
    submit = st.form_submit_button('Submit')

if submit:
    
    df_input = pd.DataFrame(form_input, index=[name])
    y_pred = model.predict(df_input)[0]
    
    st.write(f'The estimated price is ${y_pred:,.2f}')
        
    shap_values = explainer.shap_values(df_input)
    st.write('Based on the model, the influence of each feature is:')
    fp = shap.force_plot(explainer.expected_value, shap_values, df_input)
    st_shap(fp, height=300)