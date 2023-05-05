import os
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Slackに通知する関数
def send_slack_notification(message):
    client = WebClient(token=os.environ['SLACK_API_TOKEN'])  #export SLACK_API_TOKEN=api-key
    try:
        response = client.chat_postMessage(
            channel='#general',  # 通知するチャンネル
            text=message  # 通知するメッセージ
        )
        print(response)
    except SlackApiError as e:
        print(f"Error sending message: {e}")

# Pythonコードの処理が終了したことを通知する
send_slack_notification('Pythonコードの処理が終了しました。確認してください。')

