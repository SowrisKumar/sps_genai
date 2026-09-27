from fastapi import FastAPI
import spacy

app = FastAPI()

nlp = spacy.load("en_core_web_md")


@app.get("/")
def read_root():
    return {"message": "FastAPI is working"}


@app.get("/embedding")
def get_embedding(word: str):
    doc = nlp(word)

    return {
        "word": word,
        "embedding": doc.vector.tolist()
    }