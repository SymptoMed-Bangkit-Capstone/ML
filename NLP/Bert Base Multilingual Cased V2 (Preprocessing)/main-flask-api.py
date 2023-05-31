import os
import uvicorn
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

tokenizer = BertTokenizer.from_pretrained("./content/tokenizer_bert_multilungual_v2", local_files_only=True)    # sesuaikan dengan nama folder tokenizer yang sudah diupload
model = BertForSequenceClassification.from_pretrained("./content/model_bert_multilungual_v2", from_tf=True)     # sesuaikan dengan nama folder model yang sudah diupload
pipe = TextClassificationPipeline(model=model, tokenizer=tokenizer, top_k = 24)

def predict(input):
    text = str(input).lower()
    pred = pipe(text)
    kelas = pred[0][0]['label'].title()
    prob = pred[0][0]['score']
    output = "Terdiagnosa sebagai : " + str(kelas)+ ", dengan probabilitas : " +str(round((prob)*100, 2))+'%'
    return(output)

@app.get("/")
def hello_world():
    return ("hello world")

@app.post("/")
def add_item(item: Item):
    hasil = predict(item.query)
    return {hasil}

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=port, timeout_keep_alive=1200)