import streamlit as st
import requests
import os
import pandas as pd
from manim import *

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTR8fHNwYWNlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60")
    }
    .sidebar .sidebar-content {
         background: url("https://images.unsplash.com/photo-1506318137071-a8e063b4bec0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MTR8fHNwYWNlfGVufDB8fDB8fA%3D%3D&auto=format&fit=crop&w=500&q=60")
     }
    </style>
    """,
    unsafe_allow_html=True
)




st.title('Welcome to our exoplanet hunter API')

st.markdown('''<font-size= '40px'><font color='red'>**THIS TEXT WILL BE RED**</font><font-size>''', unsafe_allow_html=True)

st.subheader('please upload your lightcurve below:')

uploaded_files = st.file_uploader('Choose a CSV file', accept_multiple_files=True)
for uploaded_file in uploaded_files:
    df = pd.read_csv(uploaded_file)
    list_of_values = list(df['0'].values)
    st.write('filename:', uploaded_file.name)

#dump data into a csv file for animation.py to use
df.to_csv('animation_data.csv',index=False)

st.subheader('Your data:')
#Easy draw of the data
st.line_chart(data=df, width=300, height=200)

#url = 'https://exohunter-container-2zte5wxl7q-an.a.run.app'
url = 'http://127.0.0.1:8000/predict'

st.subheader('Result:')
#Getting result from the API
with st.spinner('Running calculation...'):
    response = requests.post(url,json={'instances': list_of_values})
    response = response.json()
st.success('Done!')
final_res = response['prediction']
final_res

st.metric(label='Confidence level', value='60%', delta=None, delta_color="normal")


if final_res == 'This star is LIKELY to have exoplanet(s)':
    st.balloons()
    st.text(
        '''⣿⣿⣿⣿⣿⡏⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⣿
⣿⣿⣿⣿⣿⣿⠀⠀⠀⠈⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⠉⠁⠀⣿
⣿⣿⣿⣿⣿⣿⣧⡀⠀⠀⠀⠀⠙⠿⠿⠿⠻⠿⠿⠟⠿⠛⠉⠀⠀⠀⠀⠀⣸⣿
⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣴⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡟⠀⠀⢰⣹⡆⠀⠀⠀⠀⠀⠀⣭⣷⠀⠀⠀⠸⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠈⠉⠀⠀⠤⠄⠀⠀⠀⠉⠁⠀⠀⠀⠀⢿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⢾⣿⣷⠀⠀⠀⠀⡠⠤⢄⠀⠀⠀⠠⣿⣿⣷⠀⢸⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡀⠉⠀⠀⠀⠀⠀⢄⠀⢀⠀⠀⠀⠀⠉⠉⠁⠀⠀⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿

⢀⣠⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⣠⣤⣶⣶
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⢰⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⣀⣀⣾⣿⣿⣿⣿'''
    )

    kepler_name  = st.text_input(label='Awesome! How would like to call your planet?', value='keplerid')

    #Run the script to make the animation
    os.system("manim --quality l --format mp4 animation.py FollowingGraphCamera")
    #Remove the partial animation
    os.system("find media/videos/animation/480p15/partial_movie_files/FollowingGraphCamera -name '*.mp4' -delete")
    #Display the animation on streamlit
    video_file = open('media/videos/animation/480p15/FollowingGraphCamera.mp4', 'rb')
    video_bytes = video_file.read()

st.title('Animation')
if os.path.isfile('media/videos/animation/480p15/FollowingGraphCamera.mp4'):
    st.video(video_bytes)
    with open("media/videos/animation/480p15/FollowingGraphCamera.mp4", "rb") as file:
        btn = st.download_button(
            label="Download video",
            data=file,
            file_name="video.mp4",
            mime="video/mp4")
else:
    st.write('Sorry, no animation for you')
