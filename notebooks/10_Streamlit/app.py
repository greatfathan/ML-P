import pandas as pd
import pickle
import streamlit as st

with open('artifacts/model.pkl', 'rb') as f:
    model = pickle.load(f)

with st.sidebar.form(key='house_price'):
    
    form_input = {
        'Bedrooms': st.number_input('Bedrooms', value=3),
        'Bathrooms': st.number_input('Bathrooms', value=2),
        'Garage': st.number_input('Garage', 2),
        'Floor Area': st.number_input('Floor Area', 200),
        'Build Year': st.number_input('Build Year', 2000),
    }
    
    name = st.text_input('Address', placeholder='Wall Street')
    submit = st.form_submit_button('Submit')

if submit:
    
    df_input = pd.DataFrame(form_input, index=[name])
    st.write('For a house with the following features:')
    st.write(df_input)
    
    y_pred = model.predict(df_input)[0]
    st.write(f'The estimated price is ${y_pred:,.2f}')