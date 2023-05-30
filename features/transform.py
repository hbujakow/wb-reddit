import pandas as pd
from transformers import pipeline
from datetime import datetime

classifier = pipeline("text-classification",
                      model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)


class Tree:
    def __init__(self, post_id, comments_df, posts_df) -> None:
        self.comments_df = comments_df[comments_df.link_id == post_id]
        self.post = posts_df[posts_df.id == post_id]
        self.structure = self._create_structure()

    def how_many_nodes(self):
        return len(self.comments_df)

    def how_many_controversial(self):
        return self.comments_df['controversiality'].sum()

    def wiener_index(self):
        comment_tree = self.structure
        n = len(comment_tree)
        if n <= 1:
            return 0
        else:
            depths = [
                self.depth(comment_tree[comment_id]) for comment_id in comment_tree
            ]
            return sum(depths) / (n * (n - 1)) + 1

    def depth(self, comment_tree=None, depth=0):
        comment_tree = (
            comment_tree if isinstance(comment_tree, dict) else self.structure
        )
        if not comment_tree:
            return depth
        depths = []
        for child_comments in comment_tree.values():
            depths.append(self.depth(child_comments, depth=depth + 1))
        return max(depths) + 1

    def _create_structure(self, parent_id=None):
        parent_id = self.post.id.values[0] if parent_id is None else parent_id
        tree = {}
        for i, row in self.comments_df[
            self.comments_df["parent_id"] == parent_id
        ].iterrows():
            comment_id = row["id"]
            tree[comment_id] = self._create_structure(comment_id)
        return tree

    def print_comment_tree(self, comment_tree=None, indent=0):
        comment_tree = (
            comment_tree if isinstance(comment_tree, dict) else self.structure
        )
        for comment_id, child_comments in comment_tree.items():
            print(" " * indent + f"Comment {comment_id}")
            if child_comments:
                self.print_comment_tree(child_comments, indent=indent + 4)


class Transformer:
    def __init__(self, posts, comments) -> None:
        self.posts = posts
        self.comments = comments

    def run(self):
        self._createfeatures()
        return self.posts, self.comments

    def find_emotion(self, text):
        classes = classifier(text)
        element_with_highest_score = sorted(
            classes[0], key=lambda x: -x['score'])[:2]
        return [x['label'] for x in element_with_highest_score]

    def calculate_date_range(self, tree, comments_df):
        def get_first_elements(tree):
            return list(tree.keys())

        def get_leaf_nodes(tree, leaves):
            for key, value in tree.items():
                if isinstance(value, dict) and len(value) > 0:
                    get_leaf_nodes(value, leaves)
                else:
                    leaves.append(key)
            return leaves

        first_elements = get_first_elements(tree)
        subset_df = comments_df[comments_df['id'].isin(first_elements)]
        first_date = subset_df['created_utc'].min()

        leaf_nodes = get_leaf_nodes(tree, [])
        subset_df = comments_df[comments_df['id'].isin(leaf_nodes)]
        last_date = subset_df['created_utc'].max()

        date_format = '%Y-%m-%dT%H:%M:%S'
        date1 = datetime.strptime(last_date, date_format)
        date2 = datetime.strptime(first_date, date_format)
        difference = (date1 - date2).total_seconds() / 3600

        return difference

    def calculate_gini_coefficient(self, tree, comments_df):
        def count_comments(tree, author_dict, comment_df):
            for comment_id, children in tree.items():
                author = comment_df.loc[comment_df['id'] == comment_id, 'author'].values
                if len(author) > 0:
                    author_dict[author[0]] = author_dict.get(author[0], 0) + 1
                count_comments(children, author_dict, comment_df)

        def gini(y):
            sorted_y = sorted(y)
            n = len(sorted_y)
            summation = sum((2 * i - n - 1) * y_i for i, y_i in enumerate(sorted_y, start=1))
            denominator = n * n * sum(sorted_y) / n
            gini_coefficient = summation / denominator
            return gini_coefficient

        author_dict = {}
        count_comments(tree, author_dict, comments_df)
        numbers_only = [value for key, value in author_dict.items() if key != '[deleted]']
        return gini(numbers_only)
    
    def _createfeatures(self):
        posts_copy = self.posts.copy()
        comments_copy = self.comments.copy()
        ids = posts_copy.id.values

        # create features for posts, filling with unknown first
        posts_copy["wiener_index"] = "unknown"
        posts_copy["depth"] = "unknown"
        posts_copy["no_comments"] = "unknown"
        posts_copy["sentiment_post"] = posts_copy["selftext"].apply(
            lambda x: self.find_emotion(x) if x != '[no_text]' else 'no_sentiment')
        posts_copy["sentiment_title"] = posts_copy["title"].apply(
            lambda x: self.find_emotion(x) if x != '[no_text]' else 'no_sentiment')

        for idd in ids:
            tree = Tree(idd, comments_copy, posts_copy)

            # update values
            posts_copy.loc[posts_copy.id == idd,
                           "wiener_index"] = tree.wiener_index()
            posts_copy.loc[posts_copy.id == idd, "post_duration"] = self.calculate_date_range(
                tree.structure, comments_copy)
            posts_copy.loc[posts_copy.id == idd, "depth"] = tree.depth()
            posts_copy.loc[posts_copy.id == idd,
                           "no_comments"] = tree.how_many_nodes()
            posts_copy.loc[posts_copy.id == idd,
                           "no_controversial"] = tree.how_many_controversial()

        self.posts, self.comments = posts_copy, comments_copy
     
    def _giniindex(self):
        posts_copy = self.posts.copy()
        comments_copy = self.comments.copy()
        ids = posts_copy.id.values
        for idd in ids:
            tree = Tree(idd, comments_copy, posts_copy)
            posts_copy.loc[posts_copy.id == idd, "gini_coefficient"] = self.calculate_gini_coefficient(
                tree.structure, comments_copy)
        self.posts, self.comments = posts_copy, comments_copy
        return self.posts, self.comments
