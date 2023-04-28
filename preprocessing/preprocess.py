import pandas as pd
from datetime import datetime
import copy
import urllib.parse

sites = [
    "facebook",
    "youtube",
    "twitter",
    "reddit",
    "instagram",
    "tumblr",
    "imgur",
    "other",
]


class Preprocessor:
    def __init__(self, posts, comments) -> None:
        self.posts = posts
        self.comments = comments

    def run(self):
        self._drop()
        self._fillna()
        self._transform()
        return self.posts, self.comments

    @staticmethod
    def _extract_domain(link):
        parsed_link = urllib.parse.urlparse(link)
        domain = parsed_link.netloc
        if domain.startswith("www."):
            domain = domain[4:]
        return domain

    @staticmethod
    def _get_site(domain):
        for site in sites:
            if site in domain:
                return site
        return "other"

    def _drop(self):
        posts_copy = copy.deepcopy(self.posts)
        comments_copy = copy.deepcopy(self.comments)
        posts_copy.drop(
            columns=["ups", "downs", "num_comments", "permalink"], inplace=True
        )
        comments_copy.drop(
            [
                "score_hidden",
                "archived",
                "name",
                "downs",
                "ups",
                "subreddit_id",
                "author_flair_css_class",
                "year_month",
                "distinguished",
            ],
            axis=1,
            inplace=True,
        )
        self.posts, self.comments = posts_copy, comments_copy

    def _fillna(self):
        posts_copy = copy.deepcopy(self.posts)
        comments_copy = copy.deepcopy(self.comments)

        posts_copy.score.fillna(0, inplace=True)
        posts_copy.selftext.fillna("[no_text]", inplace=True)
        comments_copy.body.fillna("[no_text]", inplace=True)
        comments_copy.author_flair_text.fillna("[no_text]", inplace=True)
        comments_copy.score.fillna(0, inplace=True)
        self.posts, self.comments = posts_copy, comments_copy

    def _transform(self):
        posts_copy = copy.deepcopy(self.posts)
        comments_copy = copy.deepcopy(self.comments)

        comments_copy.created_utc = comments_copy.created_utc.apply(
            lambda x: datetime.fromtimestamp(x).isoformat()
        )
        comments_copy.retrieved_on = comments_copy.retrieved_on.apply(
            lambda x: datetime.fromtimestamp(x).isoformat()
        )
        comments_copy.parent_id = comments_copy.parent_id.transform(lambda x: x[3:])
        comments_copy.link_id = comments_copy.link_id.transform(lambda x: x[3:])
        posts_copy.url = posts_copy.url.apply(
            lambda x: self._get_site(self._extract_domain(x))
        )

        self.posts, self.comments = posts_copy, comments_copy
