import pandas as pd


class Tree:
    def __init__(self, post_id, comments_df, posts_df) -> None:
        self.comments_df = comments_df[comments_df.link_id == post_id]
        self.post = posts_df[posts_df.id == post_id]
        self.structure = self._create_structure()

    def how_many_nodes(self):
        return len(self.comments_df)

    def wiener_index(self):
        comment_tree = self.structure
        n = len(comment_tree)
        if n <= 1:
            return 0
        else:
            depths = [self.depth(comment_tree[comment_id])
                      for comment_id in comment_tree]
            return sum(depths) / (n * (n-1)) + 1

    def depth(self, comment_tree=None, depth=0):
        comment_tree = comment_tree if isinstance(
            comment_tree, dict) else self.structure
        if not comment_tree:
            return depth
        depths = []
        for child_comments in comment_tree.values():
            depths.append(self.depth(child_comments, depth=depth+1))
        return max(depths) + 1

    def _create_structure(self, parent_id=None):
        parent_id = self.post.id.values[0] if parent_id is None else parent_id
        tree = {}
        for i, row in self.comments_df[self.comments_df['parent_id'] == parent_id].iterrows():
            comment_id = row['id']
            tree[comment_id] = self._create_structure(comment_id)
        return tree

    def print_comment_tree(self, comment_tree=None, indent=0):
        comment_tree = comment_tree if isinstance(
            comment_tree, dict) else self.structure
        for comment_id, child_comments in comment_tree.items():
            print(" " * indent + f"Comment {comment_id}")
            if child_comments:
                self.print_comment_tree(child_comments, indent=indent + 4)


class Transformer:
    def __init__(self, posts, comments) -> None:
        self.posts = posts
        self.comments = comments

        self._createfeatures()

    def _createfeatures(self):
        ids = self.posts.id.values

        # create features for posts, filling with unknown first
        self.posts['wiener_index'] = 'unknown'
        self.posts['depth'] = 'unknown'
        self.posts['no_comments'] = 'unknown'

        for idd in ids:
            tree = Tree(idd, self.comments, self.posts)

            # update values
            self.posts.loc[self.posts.id == idd,
                           'wiener_index'] = tree.wiener_index()
            self.posts.loc[self.posts.id == idd, 'depth'] = tree.depth()
            self.posts.loc[self.posts.id == idd,
                           'no_comments'] = tree.how_many_nodes()
