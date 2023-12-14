import streamlit as st
import requests
from PIL import Image
import pandas as pd
import numpy as np


# Call API
url = "https://prdapp-legu45dzla-ew.a.run.app"
PROCESSING_MESSAGE = "Processing..."


def call_type_api():
    print(">>> call_type_api.......")
    # response = requests.get(url).json()
    api_endpoint = ""
    # response = requests.get(api_endpoint).json()

    # print("Test API", response)
    # print(f"Test API - {response['catchability']}")

    # api_results = response["catchability"]
    # return api_results


st.set_page_config(
    page_title="My PokéApp",
    page_icon="🦈",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)

# Header

## Image
image_path = "images/pokeball.png"
image = Image.open(image_path)
st.image(image, width=50, use_column_width=False)

## Main Title
st.title(
    """
My PokéApp
"""
)

# Pokémon Type
with st.container(border=True):
    st.markdown(
        """
    ### 🩻 You want to know the type of a Pokémon.
    """
    )

    st.warning("Please take a picture, upload an image or input a url.")

    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.warning("Please take a picture, upload an image or input a url.")
    # with col3:
    #     if st.button("Compute the type", key="compute_type_btn"):
    #         pass

    cols = st.columns(2)  # without url field
    # cols = st.columns(3) # with url field
    # Caméra
    with cols[0]:
        img_file_buffer = st.camera_input("Take a picture of the Pokémon:")

        if img_file_buffer is not None:
            # Display the image user took
            st.image(
                Image.open(img_file_buffer), caption="Here's the photo you took ☝️"
            )
            with st.spinner(PROCESSING_MESSAGE):
                # Get bytes from the file buffer
                img_bytes = img_file_buffer.getvalue()

                # Make request to  API
                res = requests.post(url + "/predict_file", files={"img": img_bytes})

                dict_results = res.json()
                results = pd.DataFrame.from_dict(dict_results, orient="index")
                results.reset_index(inplace=True)
                results.columns = ["Type", "Probability"]
                print(">>> results = ", results)

                if res.status_code == 200:
                    print(f"✅ status_code = 200")
                    print("   API response = ", dict_results)
                    # Promt results
                    st.success("**The predicted types are:**", icon="🤖")
                    st.markdown(
                        results.style.hide(axis="index").to_html(),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("**Oops**, something went wrong 😓 Please try again.")
                    print(f"❌ status_code = {res.status_code}")
                    print("   API response = ", res.content)

    # Image uploadée
    with cols[1]:
        uploaded_file = st.file_uploader(
            "Upload your Pokémon image:", type=["jpg", "png", "jpeg"]
        )

        if uploaded_file is not None:
            # Display the image user uploaded
            st.image(
                Image.open(uploaded_file), caption="Here's the image you uploaded ☝️"
            )

            with st.spinner(PROCESSING_MESSAGE):
                # Get bytes from the file buffer
                img_bytes = uploaded_file.getvalue()

                # Make request to  API
                res = requests.post(url + "/predict_file", files={"img": img_bytes})

                #  Emile : for local tests
                # dict_results = {
                #     # "Type": "Probability",
                #     "Poison": "64.34%",
                #     "Water": "15.78%",
                #     "Ground": "10.79%",
                # }
                dict_results = res.json()
                results = pd.DataFrame.from_dict(dict_results, orient="index")
                results.reset_index(inplace=True)
                print(">>> results = ", results)
                results.columns = ["Type", "Probability"]
                print("... results = ", results)
                # res_array = np.array(dict_results)
                # print(">>> res_array = ", res_array)

                if res.status_code == 200:
                    # if 200 == 200: # local test
                    ### Display the image returned by the API
                    print(f"✅ status_code = 200")
                    print("   API response = ", dict_results)
                    # st.write("**The predicted types are:**")
                    # st.markdown("**Test**, et alors")
                    # Promt results
                    st.success("**The predicted types are:**", icon="🤖")
                    # st.table(results)
                    st.markdown(
                        results.style.hide(axis="index").to_html(),
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown("**Oops**, something went wrong 😓 Please try again.")
                    print(f"❌ status_code = {res.status_code}")
                    print("   API response = ", res.content)

    # with cols[2]:
    #     given_url = st.text_input("Input a url:", key="url_field")

    #     if given_url is not None:
    #         st.write(given_url)
    #         ### Display the image user uploaded
    #         given_url = (
    #             "https://archives.bulbagarden.net/media/upload/0/0c/0810Grookey.png"
    #         )
    #         st.image(given_url, caption="Here's the image you input ☝️")

    #         #  a metter qq part pour rester le champ
    #         # st.session_state["url_field"] = ""
