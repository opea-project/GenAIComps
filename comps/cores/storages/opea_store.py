from .arangodb import ArangoDBStore
# from .redisdb import RedisDBStore
# from .mongodb import MongoDBStore

def opea_store(name: str, *args, **kwargs):
    if name == "arangodb":
        return ArangoDBStore(name, *args, **kwargs)
    # elif name == "redis":
    #     return RedisDBStore(*args, **kwargs)
    # elif name == "mongodb":
    #     return MongoDBStore(*args, **kwargs)
    else:
        raise ValueError(f"Unknown Data Store: {name}")
