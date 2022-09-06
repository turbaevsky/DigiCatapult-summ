FROM python:3.9.12
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY app.py .
COPY summ.py .
EXPOSE 8501
#ENTRYPOINT [ "streamlit", "run"]
#CMD ["app.py"]
CMD streamlit run app.py
