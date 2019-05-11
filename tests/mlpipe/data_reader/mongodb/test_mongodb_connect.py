"""
Expects a running local MongoDB with no authentication to connect
"""
import pytest
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from mlpipe.data_reader.mongodb import MongoDBConnect
from tests.mlpipe._setups.mongo_db_setup import cp, client_name


def test_reset():
    mongo_con = MongoDBConnect()
    mongo_con.add_connections_from_config(cp)
    mongo_con.reset_connections()
    assert len(mongo_con._connections) == 0


def test_mongodb_connection():
    mongo_con = MongoDBConnect()
    mongo_con.add_connections_from_config(cp)
    assert len(mongo_con._connections) == 1


def test_get_client_success():
    mongo_con = MongoDBConnect()
    mongo_con.reset_connections()
    mongo_con.add_connections_from_config(cp)
    client = mongo_con.get_client(client_name)
    assert isinstance(client, MongoClient)


@pytest.mark.xfail(raises=ValueError)
def test_get_client_fail():
    mongo_con = MongoDBConnect()
    mongo_con.reset_connections()
    mongo_con.add_connections_from_config(cp)
    _ = mongo_con.get_client("this_does_not_exist")


def test_get_db():
    mongo_con = MongoDBConnect()
    mongo_con.reset_connections()
    mongo_con.add_connections_from_config(cp)
    client = mongo_con.get_db(client_name, "test_db")
    assert isinstance(client, Database)


def test_get_collection():
    mongo_con = MongoDBConnect()
    mongo_con.reset_connections()
    mongo_con.add_connections_from_config(cp)
    client = mongo_con.get_collection(client_name, "test_db", "test_collection")
    assert isinstance(client, Collection)
