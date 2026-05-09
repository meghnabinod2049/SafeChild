from flask import Flask, request, jsonify, send_file
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

app = Flask(__name__)

# ==================== MODEL SETUP ====================

MODEL_NAME = "distilbert-base-uncased"

LABEL_NAMES = [
    'trust',
    'isolation',
    'inappropriate',
    'secrecy',
    'solicitation'
]

# ==================== MODEL CLASS ====================

class DistilBERTMultiLabel(nn.Module):

    def __init__(self):

        super().__init__()

        self.distilbert = AutoModel.from_pretrained(
            MODEL_NAME
        )

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            768,
            5
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_state = outputs.last_hidden_state

        pooled = hidden_state[:, 0]

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits

# ==================== LOAD MODEL ====================

print("Loading model...")

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

tokenizer = AutoTokenizer.from_pretrained(
    'safechild_tokenizer'
)

model = DistilBERTMultiLabel().to(device)

model.load_state_dict(
    torch.load(
        'best_safechild_model.pth',
        map_location=device
    )
)

model.eval()

print(f"Model loaded on {device}")

# ==================== DASHBOARD ====================

@app.route('/')
def dashboard():

    return send_file(
        'safechild_dashboard.html'
    )

# ==================== PREDICTION API ====================

@app.route('/predict', methods=['POST'])
def predict():

    data = request.get_json()

    text = data.get(
        'text',
        ''
    )

    # ==================== TOKENIZATION ====================

    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)

    attention_mask = encoding['attention_mask'].to(device)

    # ==================== MODEL PREDICTION ====================

    with torch.no_grad():

        outputs = model(
            input_ids,
            attention_mask
        )

        probs = torch.sigmoid(
            outputs
        ).cpu().numpy()[0]

    # ==================== CATEGORY SCORES ====================

    trust = float(probs[0])

    isolation = float(probs[1])

    inappropriate = float(probs[2])

    secrecy = float(probs[3])

    solicitation = float(probs[4])

    text_lower = text.lower()

    # ==================== TRUST BUILDING ====================

    trust_words = [

        "special",
        "understand you",
        "mature for your age",
        "you are mature",
        "only you",
        "i care about you",
        "spoil you",
        "you make me happy",
        "i enjoy talking to you"
    ]

    # ==================== ISOLATION ====================

    isolation_words = [

        "parents dont understand",
        "parents don't understand",
        "nobody understands",
        "only talk to me",
        "they will judge you",
        "dont mention me",
        "don't mention me",
        "your parents are strict"
    ]

    # ==================== SECRECY ====================

    secrecy_words = [

        "don't tell",
        "dont tell",
        "keep this secret",
        "between us",
        "delete messages",
        "delete the chat",
        "hide this",
        "nobody will know",
        "parents see",
        "keep this between us"
    ]

    # ==================== SOLICITATION ====================

    solicitation_words = [

        "send pics",
        "send pic",
        "private picture",
        "private pics",
        "hot pics",
        "send photos",
        "meet alone",
        "come alone",
        "video call alone",
        "meet at night",
        "abandoned place",
        "send me a picture",
        "private photo"
    ]

    # ==================== APPLY BOOSTS ====================

    for word in trust_words:

        if word in text_lower:

            trust += 0.35

    for word in isolation_words:

        if word in text_lower:

            isolation += 0.45

    for word in secrecy_words:

        if word in text_lower:

            secrecy += 0.55

    for word in solicitation_words:

        if word in text_lower:

            solicitation += 0.75

            inappropriate += 0.40

    # ==================== LIMIT VALUES ====================

    trust = min(trust, 1.0)

    isolation = min(isolation, 1.0)

    inappropriate = min(inappropriate, 1.0)

    secrecy = min(secrecy, 1.0)

    solicitation = min(solicitation, 1.0)

    # ==================== OVERALL RISK ====================

    overall_risk = (

        trust * 0.15 +

        isolation * 0.20 +

        inappropriate * 0.20 +

        secrecy * 0.20 +

        solicitation * 0.25
    )

    overall_risk = min(
        overall_risk,
        1.0
    )

    # ==================== ALERT LEVEL ====================

    if overall_risk >= 0.75:

        alert_level = "CRITICAL"

    elif overall_risk >= 0.50:

        alert_level = "HIGH"

    elif overall_risk >= 0.25:

        alert_level = "MEDIUM"

    else:

        alert_level = "LOW"

    # ==================== RESPONSE ====================

    return jsonify({

        'trust': trust,

        'isolation': isolation,

        'inappropriate': inappropriate,

        'secrecy': secrecy,

        'solicitation': solicitation,

        'overall_risk': overall_risk,

        'alert_level': alert_level
    })

# ==================== HEALTH CHECK ====================

@app.route('/health', methods=['GET'])
def health():

    return jsonify({

        'status': 'healthy',

        'model': 'DistilBERT',

        'device': str(device)
    })

# ==================== RUN APP ====================

if __name__ == '__main__':

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )