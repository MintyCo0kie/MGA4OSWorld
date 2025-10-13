from google_auth_oauthlib.flow import InstalledAppFlow
import json

SCOPES = ['https://www.googleapis.com/auth/drive']

flow = InstalledAppFlow.from_client_secrets_file(
    'client_secrets.json',
    scopes=SCOPES
)

credentials = flow.run_local_server(port=0)

with open('credentials.json', 'w') as f:
    f.write(credentials.to_json())

print("OAuth2 credentials have been saved to credentials.json.")