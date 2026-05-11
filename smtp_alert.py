import smtplib

from email.mime.text import MIMEText

from email.mime.multipart import MIMEMultipart


SENDER_EMAIL = "meghnabinod2020@gmail.com"

APP_PASSWORD = "kkhy ndkc ipzb lykg"

RECEIVER_EMAIL = "meghnabinod2020@gmail.com"


def send_alert_email(message, risk):

    try:

        subject = "🚨 SafeChild Critical Grooming Alert"

        body = f"""

SAFECHILD ALERT SYSTEM

Critical grooming-risk conversation detected.

Risk Score: {risk:.2f}

Flagged Message:
{message}

Immediate moderator review recommended.

"""

        msg = MIMEMultipart()

        msg["From"] = SENDER_EMAIL

        msg["To"] = RECEIVER_EMAIL

        msg["Subject"] = subject

        msg.attach(

            MIMEText(body, "plain")
        )

        server = smtplib.SMTP(
            "smtp.gmail.com",
            587
        )

        server.starttls()

        server.login(
            SENDER_EMAIL,
            APP_PASSWORD
        )

        server.sendmail(

            SENDER_EMAIL,

            RECEIVER_EMAIL,

            msg.as_string()
        )

        server.quit()

        print("EMAIL ALERT SENT SUCCESSFULLY")

    except Exception as e:

        print("EMAIL ALERT ERROR:", e)