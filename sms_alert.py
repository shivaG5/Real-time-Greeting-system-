from twilio.rest import Client
from dotenv import load_dotenv
import os
load_dotenv()

account_sid=os.getenv("TWILIO_ACCOUNT_SID")
auth_token=os.getenv("TWILIO_ACCOUNT_TOKEN")
twilio_number=os.getenv("TWILIO_PHONE")
my_number=os.getenv("MY_PHONE")

client=Client(account_sid,auth_token)
def send_alert_sms(message_text):
    message=client.messages.create(
        body=message_text,
        from_=twilio_number,
        to=my_number
    )
    print("Alert sent!SID:",message.sid)