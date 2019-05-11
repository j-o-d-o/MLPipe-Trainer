import configparser
import json

client_name = "localhost_mongo_db"

config_file_path = "./tests/mlpipe/_setups/config.ini"
cp = configparser.ConfigParser()
flag = cp.read(config_file_path)
if len(flag) == 0:
    raise ValueError("path to config.ini is wrong!")

json_path_name = 'tests/mlpipe/_setups/test_documents.json'
docs = json.load(open(json_path_name))
