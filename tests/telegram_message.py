import requests

BOT_TOKEN = "Redacted"
CHAT_ID = -1004460895066

url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

response = requests.post(
    url,
    data={
        "chat_id": CHAT_ID,
        "text": "🤖\n\nBot berhasil mengirim pesan ke channel!",
    }
)

print(response.json())