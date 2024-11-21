from retriever.api_keys import PINECONE_API_KEY
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

def get_pinecone_client():
    return Pinecone(api_key=PINECONE_API_KEY)

METRIC_COSINE = "cosine"
METRIC_L2 = "euclidean"

def get_or_create_index(pc, index_name, dims, metric=METRIC_L2):
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dims, 
            metric=metric, # Replace with your model metric
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ) 
        )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)
    return pc.Index(index_name)
