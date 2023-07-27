# book_schema.py
from mongoengine import Document, StringField, DateField, IntField

class Book(Document):
    title = StringField(required=True)
    author = StringField(required=True)
    publication_date = DateField(default=Date.now)
    pages = IntField(required=True)
    genre = StringField(required=True)
