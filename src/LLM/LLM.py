import torch

import os
from openai import OpenAI

# 配置代理
os.environ["http_proxy"] = "http://10.61.2.90:1082"
os.environ["https_proxy"] = "http://10.61.2.90:1082"


class LLM_model(torch.nn.Module):
    def __init__(self, hid_dim, dataset, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.dataset = dataset
        self.device = device
        self.client = OpenAI()
        self.chat_model_name = "gpt-4o-mini"
        self.emb_model_name = "text-embedding-3-large"

    def graph_encode(self):
        if "thgl-github" in self.dataset:
            text = "thgl-github is a continuous time-series dynamic heterogeneous graph dataset that records software development interaction activities on the github website. Its time division is composed of interaction timestamps. The dataset is 10,000 interactions on January 1, 2014. There are 4 types of nodes: user, issue, repository, and PullRequest. There are 14 types of edges: Issue belongs to Repository, User closes Issue, Issue closed in Repository, User opens Issue, User reopens Issue, Issue reopened in Repository, User opens Pull Request, Pull Request belongs to Repository, User closes Pull Request, Pull Request closed in Repository, User reopens Pull Request, Pull Request reopened in Repository, User added to Repository, Repository forks Repository."
        elif "thgl-forum" in self.dataset:
            text = "thgl-forum is a real-world temporal heterogeneous graph from the TGB 2.0 benchmark, which documents user interactions on Reddit. The dataset captures activities from January 2014. It is composed of two types of nodes ( user and subreddit) and two primary types of relations: user-reply-user and user-post-subreddit."
        elif "thgl-software" in self.dataset:
            text = "thgl-software is a continuous time-series dynamic heterogeneous graph dataset that records software development interaction activities on the github website. Its time division is composed of interaction timestamps. The dataset is 10,000 interactions on January 1, 2014. There are 4 types of nodes: user, issue, repository, and PullRequest. There are 14 types of edges: Issue belongs to Repository, User closes Issue, Issue closed in Repository, User opens Issue, User reopens Issue, Issue reopened in Repository, User opens Pull Request, Pull Request belongs to Repository, User closes Pull Request, Pull Request closed in Repository, User reopens Pull Request, Pull Request reopened in Repository, User added to Repository, Repository forks Repository."
        elif "thgl-myket" in self.dataset:
            text = "thgl-myket is a real-world temporal heterogeneous graph from Myket, the third largest Android application market in Iran. This dataset records the interaction between users and applications (Apps). It contains two types of nodes (users and apps) and two main types of relationships: user installs an app and user updates an app."
        request = "Please output a summary of the information about this temporal heterogeneous graph in the following format: {Domain:,Introduction:,Node types:,Relations:}."
        summary = self.client.responses.create(
            model=self.chat_model_name,
            input=text + request,
        ).output_text
        # print(summary)
        emb = (
            self.client.embeddings.create(
                model=self.emb_model_name,
                input=summary,
                dimensions=self.hid_dim,
            )
            .data[0]
            .embedding
        )
        emb = torch.tensor(emb)
        emb = emb.to(self.device)
        return emb

    def cate_encode(self):
        if "thgl-github" in self.dataset:
            text = [
                "Issues within a repository that can be opened, closed, or reopened by users.",
                "Repositories on GitHub that contain issues and pull requests, can be forked, and can have users added to them.",
                "Users on GitHub who can open, close, or reopen issues and pull requests, and can be added to repositories.",
                "Pull Requests within a repository that can be opened, closed, or reopened by users.",
            ]
        elif "thgl-forum" in self.dataset:
            text = [
                "Reddit users who may reply to other users or post in subreddits.",
                "Reddit subreddits that may be posted in by users.",
            ]
        elif "thgl-software" in self.dataset:
            text = [
                "Issues within a repository that can be opened, closed, or reopened by users.",
                "Repositories on GitHub that contain issues and pull requests, can be forked, and can have users added to them.",
                "Users on GitHub who can open, close, or reopen issues and pull requests, and can be added to repositories.",
                "Pull Requests within a repository that can be opened, closed, or reopened by users.",
            ]
        elif "thgl-myket" in self.dataset:
            text = [
                "Myket users who may install or upgrade apps.",
                "Myket apps that may be installed or upgraded by users.",
            ]
        request = "Please output a summary of the information about this node type in the following format: {Introduction:,Relevant relations:}."
        embeddings = []

        for t in text:
            summary = self.client.responses.create(
                model=self.chat_model_name,
                input=t + request,
            ).output_text
            # print(summary)

            emb = (
                self.client.embeddings.create(
                    model=self.emb_model_name,
                    input=summary,
                    dimensions=self.hid_dim,
                )
                .data[0]
                .embedding
            )
            emb = torch.tensor(emb)
            emb = emb.to(self.device)
            embeddings.append(emb)

        return embeddings


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # dataset = "thgl-github-subset"
    # dataset = "thgl-software-subset"
    # dataset = "thgl-forum-subset"
    # dataset = "thgl-myket-subset"
    embedding_dim = 512
    for dataset in [
        "thgl-github-subset",
        "thgl-software-subset",
        "thgl-forum-subset",
        "thgl-myket-subset",
    ]:
        model = LLM_model(embedding_dim, dataset, device)
        emb_graph = model.graph_encode()
        emb_cate = model.cate_encode()
        save_path_graph = os.path.join(
            "tgb/DATA",
            dataset.replace("-", "_"),
            dataset + f"_emb{embedding_dim}_graph.pt",
        )
        save_path_cate = os.path.join(
            "tgb/DATA",
            dataset.replace("-", "_"),
            dataset + f"_emb{embedding_dim}_cate.pt",
        )
        torch.save(emb_graph, save_path_graph)
        torch.save(emb_cate, save_path_cate)
