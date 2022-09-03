import streamlit as st
import os
import json

from summ import Summ

st.set_page_config(
    page_title="Extractor",
    page_icon="",
)

st.title("Extractor")
st.header("")

with st.expander("About this app", expanded=True):

    st.write(
        """     
        - Currently, there is a wide variety of  Natural Language Processing (NLP) tasks; these range from a very basic pattern search to profoundly intelligent chat bots with the functionality to replace human operators.
        - In particular, there is a specific set of tasks which enables us to process an especially  large volume of papers - either in pdf or html format – in order to extract the most important information. Additionally,  it is sometimes necessary to remove certain information from the text to follow GDPR or other regulations.
	"""
    )

    st.markdown("")

st.markdown("")
st.markdown("## **📌 Paste document **")

with st.sidebar:
    ModelType = st.radio(
        "Choose your summarisation model",
        ["TF-IDF extractive", "transformes (abstractive)", "PEGASUS (abstractive)"],
        help="At present, you can choose between 3 models to play with your text. More to come!",
        )

    max_length = st.slider(
        "max number of words for transformers",
        min_value=10,
        max_value=500,
        value=100,
        help="It works for transformers abstractive summarisation only",
        )

with st.form(key='form'):
    text = st.text_area(
    "Paste your text below (max 500 words)",
    height=300,
        )

    MAX_WORDS = 500
    import re
    res = len(re.findall(r"\w+", text))
    if res > MAX_WORDS:
        st.warning(
            "⚠️ Your text contains "
            + str(res)
            + " words."
            + " Only the first 500 words will be reviewed. Stay tuned as increased allowance is coming"
        )

    text = text[:MAX_WORDS]

    exp_summ = st.text_area(
            "Paste your expected summarisation text below (optional)",
            height=100,
        )

    exp_gen = st.text_area(
            "Paste your text to be removed from the source (optional)",
            height=100,
        )

    submit_button = st.form_submit_button(label="Extract")


if not submit_button:
    st.stop()

if exp_summ == '':
    exp_summ = None
if exp_gen == '':
    exp_gen = None

t = Summ(text, exp_summ, exp_gen)

if ModelType == "TF-IDF extractive":
    summ = t.ext_summ()

elif ModelType == "transformes (abstractive)":
    summ = t.abs_summ_transf()

else:
    summ = t.abs_summ_pegasus()

# extraction        
ext = t.generalisation()

# metrics
sim = t.similarity(summ)
bl = t.bleu_score(ext)

print(f'metrics: similarity {sim}, bleu {bl}')
            
st.markdown("## Check results ")

st.subheader('Source text:')
st.write(text)
st.subheader('Summarisation:')
st.write(summ)
st.subheader('Removed sentences:')
st.write(ext)

st.header("Metrics")
col1, col2 = st.columns(2)

col1.metric('Similarity', sim)
col2.metric('BLEU', bl)
