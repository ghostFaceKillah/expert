"""
Adapted from:
https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
"""
import os
import requests
import tarfile
import tqdm

import constants


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = _get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    _save_response_content(response, destination)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None


def _save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in tqdm.tqdm(response.iter_content(CHUNK_SIZE)):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def untar(what, to_where):
    tar = tarfile.open(what)
    tar.extractall(path=to_where)
    tar.close()


def download_and_untar(what, to_where):
    temp_tar_name = 'data.tar.gz'

    print("Starting downloading expert data file from Google Drive...")
    print("Please be patient. For example, montezuma's revenge data weighs 756 MB.")
    download_file_from_google_drive(what, temp_tar_name)
    print("Done!")

    print("Un-tarring the files.")
    untar(temp_tar_name, to_where)
    print("Done!")

    print("Cleaning up...")
    os.unlink(temp_tar_name)
    print("Done!")





