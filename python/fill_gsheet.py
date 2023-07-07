from __future__ import print_function

import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SPREADSHEET_ID = '1yN0hX6dvBykjH2alkL-RB4YYNF7_aJVJYMGTK0_ynug'
CALIBRATE_RANGE_NAME = 'Calibrate Stats!A2:FX'
OUTPUT_FILE = '../output/output_calibrate.txt'
JSON_DIR = '../json/'

def main():

    with open(OUTPUT_FILE) as f:
        lines = f.readlines()
    data = []
    for line in lines : 
        line = line.replace(".",",")
        line = line.replace("\n","")
        data.append(line.split(" "))

    creds = None
    if os.path.exists(f'{JSON_DIR}token.json'):
        creds = Credentials.from_authorized_user_file(f'{JSON_DIR}token.json', SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                f'{JSON_DIR}credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open(f'{JSON_DIR}token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('sheets', 'v4', credentials=creds)
        sheet = service.spreadsheets()

        clear_values_request_body = {}
        sheet.values().clear(spreadsheetId=SPREADSHEET_ID, range='Calibrate Stats!A2:F1000', body=clear_values_request_body).execute()

        body = {'values': data}
        service.spreadsheets().values().update(
            spreadsheetId=SPREADSHEET_ID, range=CALIBRATE_RANGE_NAME.replace("X",f"{len(lines)+1}"),
            valueInputOption="USER_ENTERED", body=body).execute()

    except HttpError as err:
        print(err)


if __name__ == '__main__':
    main()