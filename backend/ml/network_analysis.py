import random
import networkx as nx
from backend.ml.base import BaseModelInterface

class RealNetworkAnalysisModel(BaseModelInterface):
    """
    Real Network Analysis using NetworkX.
    Since we don't have access to the full social graph (thousands of users),
    we generate a synthetic ego-graph centered around the target profile based
    on their follower/following counts, and calculate network centrality metrics
    to estimate the risk of being part of a bot-ring.
    """
    def __init__(self):
        self.model_name = "NetworkX-GraphAnalyzer"

    def predict(self, features: dict) -> float:
        """
        Returns a network anomaly score (0.0 to 1.0).
        """
        followers = features.get("followers", 1)
        following = features.get("following", 0)
        
        # If the account is completely isolated or too small to matter
        if followers < 5 and following < 5:
            return 0.1
            
        # 1. Generate Synthetic Ego-Graph
        # We cap the graph size for performance reasons in this MVP
        n_followers = min(followers, 200)
        n_following = min(following, 200)
        total_nodes = n_followers + n_following + 1
        
        try:
            # Create a directed graph
            G = nx.DiGraph()
            
            # Add ego node (the user)
            ego = "TARGET"
            G.add_node(ego)
            
            # Add followers (directed edges towards ego)
            for i in range(n_followers):
                node = f"follower_{i}"
                G.add_edge(node, ego)
                
            # Add following (directed edges from ego)
            for i in range(n_following):
                node = f"following_{i}"
                G.add_edge(ego, node)
                
            # Simulate bot-ring behavior: 
            # If following is high and followers is very low, or they perfectly match,
            # we inject dense reciprocal connections (cliques) to simulate a follow-train/bot-ring.
            if following > 50 and (followers < following * 0.1 or followers > following * 0.9):
                # Inject dense connections between 'following' nodes
                for i in range(min(n_following, 50)):
                    for j in range(i + 1, min(n_following, 50)):
                        if random.random() < 0.6:  # High probability of reciprocal follow
                            G.add_edge(f"following_{i}", f"following_{j}")
                            G.add_edge(f"following_{j}", f"following_{i}")
            else:
                # Normal organic network (sparse connections between followers/following)
                for i in range(min(n_following, 20)):
                    for j in range(i + 1, min(n_following, 20)):
                        if random.random() < 0.05:  # Low probability
                            G.add_edge(f"following_{i}", f"following_{j}")
                            
            # 2. Compute Graph Metrics
            # Clustering coefficient for directed graph requires conversion to undirected for standard metric
            G_undirected = G.to_undirected()
            try:
                ego_clustering = nx.clustering(G_undirected, ego)
            except:
                ego_clustering = 0.0
                
            # In-degree centrality
            in_degree_cent = nx.in_degree_centrality(G)
            ego_in_degree = in_degree_cent.get(ego, 0.0)
            
            # Out-degree centrality
            out_degree_cent = nx.out_degree_centrality(G)
            ego_out_degree = out_degree_cent.get(ego, 0.0)
            
            # 3. Calculate Risk Score based on topologies
            risk_score = 0.1
            
            # High out-degree and low in-degree is classic spammer
            if ego_out_degree > 0.5 and ego_in_degree < 0.05:
                risk_score += 0.4
                
            # Extremely high clustering in the ego network implies a dense bot-ring
            if ego_clustering > 0.4:
                risk_score += 0.4
                
            return min(round(risk_score, 2), 1.0)
            
        except Exception as e:
            print(f"NetworkX analysis error: {e}")
            return 0.5
