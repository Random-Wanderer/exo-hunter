import streamlit as st
import requests
import os
import pandas as pd

# def file_selector(folder_path='.'):
#     filenames = os.listdir(folder_path)
#     selected_filename = st.selectbox('Select a file', filenames)
#     return os.path.join(folder_path, selected_filename)

# filename = file_selector()
# st.write('You selected `%s`' % filename)


uploaded_files = st.file_uploader('Choose a CSV file', accept_multiple_files=True)
for uploaded_file in uploaded_files:
    temp = pd.read_csv(uploaded_file)
    temp = list(temp.values)
    # bytes_data = uploaded_file.read()
    # print(bytes_data)


st.write('filename:', uploaded_file.name)
st.write(temp)



url = 'https://exohunter-container-2zte5wxl7q-an.a.run.app'

#2. Let's build save the input file


#3. Let's call our API using the `requests` package...

response = requests.post(url,{'instances': temp})
response = response.json()

#4. Let's retrieve the prediction from the **JSON** returned by the API...

## Finally, we can display the prediction to the user
final_res = response
final_res
