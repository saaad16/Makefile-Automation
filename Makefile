setup:
	py -m pip install -r requirements.txt

download-data:
	py scripts\download_data.py

preprocess:
	py scripts\preprocess.py