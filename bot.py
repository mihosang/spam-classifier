import asyncio
import os
import time

from sklearn.externals import joblib
from slackclient import SlackClient

RUNNING = True

def stop():
    global RUNNING
    RUNNING = False
    print("stopping slack bot...")

class Bot(object):
    clf = joblib.load('./data/C-SVC-model.pkl')
    vect = joblib.load('vectorizer.joblib')

    def __init__(self):
        self.slack_token = os.environ["SLACK_API_TOKEN"]
        self.slack_client = SlackClient(self.slack_token)

    async def parse_bot_commands(self, slack_events):
        for event in slack_events:
            print(event)
            if not "type" in event:
                return
            if event['type'] == 'message' and not "subtype" in event:
                # get the user_id and the text of the post
                user_id, text, channel = event['user'], event['text'], event['channel']
                print(user_id, text, channel)
                x_test = self.vect.transform([text])
                predict = self.clf.predict(x_test)
                predict_proba = self.clf.predict_proba(x_test)[0][0]
                reply_message = "피싱 같습니다 " if predict == "phishing" else "정상 대화 같네요 "
                reply_message += "[예측확률 : %.2f" % float(predict_proba * 100) + "%]"
                self.slack_client.rtm_send_message(channel, reply_message)


    async def listen(self):
        if self.slack_client.rtm_connect(with_team_state=False):
            self.bot_id = self.slack_client.api_call("auth.test")["user_id"]
            print("[%s] Successfully connected, listening for commands" % self.bot_id)
            while self.slack_client.server.connected is True:
                recv_message = self.slack_client.rtm_read()
                await self.parse_bot_commands(recv_message)
                time.sleep(1)
        else:
            exit("Error, Connection Failed")

if __name__ == "__main__":
    bot = Bot()
    loop = asyncio.get_event_loop()

    loop.run_until_complete(bot.listen())
    loop.run_forever()