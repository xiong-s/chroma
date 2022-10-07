class Embedding:
    def __init__(self, data):
        self.data = data["embeddings"]
        self.width = len(self.data)
        self.inference = data["inferences"][0]
        self.labels = data["labels"]
        self.metadata = data["metadata"]
        self.resource_uri = data["resource_uri"]
        self.model = "model1"

    def __repr__(self):
        return f"Embedding<{self.resource_uri}, {self.data[:3]}... ({len(self.data)})>"
