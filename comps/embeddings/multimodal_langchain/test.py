# from multimodal_embeddings.bridgetower_embedding import BridgeTowerEmbedding
from multimodal_embeddings import BridgeTowerEmbedding
t1 = BridgeTowerEmbedding(device='hpu')
print(t1.device)