from google.cloud import language_v1

client = language_v1.LanguageServiceClient()

texts = [
    "You are very mature for your age.",
    "Don't tell your parents about our chats.",
    "Send me a picture nobody has seen.",
    "Your parents don't understand you like I do.",
    "Hope you have a great day!"
]

for text in texts:

    document = language_v1.Document(
        content=text,
        type_=language_v1.Document.Type.PLAIN_TEXT
    )

    response = client.analyze_sentiment(
        request={"document": document}
    )

    sentiment = response.document_sentiment

    print("=" * 60)
    print("TEXT:", text)
    print("Sentiment score:", sentiment.score)
    print("Sentiment magnitude:", sentiment.magnitude)