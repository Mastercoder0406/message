from twilio.rest import Client
import os
# twilio messaging format
account_sid=os.environ('account_sid') # verification of the twilio account
auth_token=os.environ('auth_token')

client=Client(account_sid,auth_token)


numbers=[] # use the standard format of +91
message=client.messages.create(
    body='Danger Detected',
    from_="+18288090892",
    to=''# You can use group of numbers from numbers list 
)
# free accounts cannot sent msg to unverifed numbers

print("DOne")