# ==================== alert_function/main.py ====================

import json

def alert_pipeline(request):

    request_json = request.get_json(silent=True)

    if request_json:

        message = request_json.get("message")

        risk = request_json.get("risk")

        moderator = request_json.get("moderator")

        print("===== MODERATOR ALERT =====")

        print("Moderator:", moderator)

        print("Risk:", risk)

        print("Message:", message)

        return {

            "status": "ALERT_TRIGGERED",

            "moderator": moderator,

            "risk": risk,

            "message": message
        }

    return {

        "status": "NO_DATA"
    }