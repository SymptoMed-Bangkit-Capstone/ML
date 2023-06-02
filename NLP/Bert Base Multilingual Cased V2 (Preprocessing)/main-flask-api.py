import uvicorn
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import TextClassificationPipeline
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()  # create a new FastAPI app instance
# port = int(os.getenv("PORT"))
port = 8080

# Define a Pydantic model for an item
class Item(BaseModel):
    query:str

model = BertForSequenceClassification.from_pretrained("./content/model_bert_multilungual_v2", from_tf=True)     # sesuaikan dengan nama folder model yang sudah diupload
tokenizer = BertTokenizer.from_pretrained("./content/tokenizer_bert_multilungual_v2", local_files_only=True)    # sesuaikan dengan nama folder tokenizer yang sudah diupload
data_rekomendasi = pd.read_csv('./Dataset Rekomendasi Hasil Prediksi NLP.csv', sep=';')                         # sesuaikan dengan nama file data rekomendasi yang sudah diupload
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k = 24)

def predict(input):
    text = str(input).lower()
    pred = pipe(text)
    kelas = pred[0][0]['label'].title()
    prob = str(round((pred[0][0]['score'])*100, 2))+'%'
    return(kelas, prob)

@app.get("/")
def main_page():
    return (
        "Selamat datang di FastAPI untuk klasifikasi teks SymptoMed. Silahkan gunakan metode POST untuk mengirimkan data. "
    )

@app.post("/")
def add_item(item: Item):
    global data_rekomendasi

    hasil, probability = predict(item.query)

    if hasil == "Pembuluh Mekar" :
        hasil = "Varises"
    elif hasil == "Spondylosis Serviks" :
        hasil = "Spondylosis"
    elif hasil == "Wasir Dimorfik" :
        hasil = "Wasir"
    
    indeks_hasil = data_rekomendasi[data_rekomendasi['Symptom'] == hasil.lower()]
    link = indeks_hasil['Detail'].values[0]
    saran = indeks_hasil['Saran'].values[0]

    return {
        "Kelas": hasil,
        "Proabilitas": probability,
        "link": link,
        "Rekomendasi": saran
    }

if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=port, timeout_keep_alive=1200)