from telethon.sync import TelegramClient

# Use your own values from my.telegram.org
api_id = 0000# use your signin method to activate your Client instance 
api_hash = ''

# The first parameter is the .session file name (absolute paths allowed)
with TelegramClient('anon', api_id, api_hash) as client:
    client.start()
    client.send_message('', 'The message you want to send to your user')

    # the first argument is the number and the second is the message you want to send to user 

#refer telethon documentation for more datails =>https://docs.telethon.dev/en/stable/basic/quick-start.html