



class CountsToCaetory:
    def __init__(self, adata, embedder, classifier):
        self.adata = adata
        self.embedder = embedder
        self.classifier = classifier

    def __call__(self, category):
        return self.counts[category]

    def __getitem__(self, category):
        return self.counts[category]