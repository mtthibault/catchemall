import os
import streamlit as st


GAR_MEMORY = os.environ.get("GAR_MEMORY")
PREDICT_API_URL = os.environ.get("PREDICT_API_URL")
print("PREDICT_API_URL = ", PREDICT_API_URL)

st.write(f"PREDICT_API_URL = {PREDICT_API_URL}")


def display_input_row(index):
    left, middle, right = st.columns(3)
    left.text_input("First", key=f"first_{index}")
    middle.text_input("Middle", key=f"middle_{index}")
    right.text_input("Last", key=f"last_{index}")


if "rows" not in st.session_state:
    st.session_state["rows"] = 0


def increase_rows():
    st.session_state["rows"] += 1


st.button("Add person", on_click=increase_rows)

for i in range(st.session_state["rows"]):
    display_input_row(i)

# Show the results
st.subheader("People")
for i in range(st.session_state["rows"]):
    st.write(
        f"Person {i+1}:",
        st.session_state[f"first_{i}"],
        st.session_state[f"middle_{i}"],
        st.session_state[f"last_{i}"],
    )


# st.selectbox
option = st.selectbox(
    "How would you like to be contacted?", ("Email", "Home phone", "Mobile phone")
)
st.write("You selected:", option)


# Severals sliders template
cols = st.columns([2, 1, 2])
minimum = cols[0].number_input("Minimum", 1, 5)
maximum = cols[2].number_input("Maximum", 6, 10, 10)
st.slider("No default, no key", minimum, maximum)
st.slider("No default, with key", minimum, maximum, key="a")
st.slider("With default, no key", minimum, maximum, value=5)
st.slider("With default, with key", minimum, maximum, value=5, key="b")

# Test
# def change_name(name):
#     st.session_state["name"] = name
# st.header(st.session_state["name"])
# st.button("Jane", on_click=change_name, args=["Jane Doe"])
# st.button("John", on_click=change_name, args=["John Doe"])
# st.header(st.session_state["name"])
