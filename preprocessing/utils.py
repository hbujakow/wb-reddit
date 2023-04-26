import pandas as pd
from datetime import datetime


class Preprocessor:
    def __init__(self, posts, comments) -> None:
        self.posts = posts
        self.comments = comments
        self._drop()
        self._fillna()
        self._transform()

    def _drop(self):
        self.posts.drop(
            columns=['ups', 'downs', 'num_comments', 'permalink'], inplace=True)
        self.comments.drop(['score_hidden', 'archived', 'name', 'downs', 'ups', 'subreddit_id',
                            'author_flair_css_class', 'year_month', 'distinguished'], axis=1, inplace=True)

    def _fillna(self):
        self.posts.score.fillna(0, inplace=True)
        self.posts.selftext.fillna('[no_text]', inplace=True)
        self.comments.body.fillna('[no_text]', inplace=True)
        self.comments.author_flair_text.fillna('[no_text]', inplace=True)
        self.comments.score.fillna(0, inplace=True)

    def _transform(self):
        self.comments.created_utc = self.comments.created_utc.apply(
            lambda x: datetime.fromtimestamp(x).isoformat())
        self.comments.retrieved_on = self.comments.retrieved_on.apply(
            lambda x: datetime.fromtimestamp(x).isoformat())
