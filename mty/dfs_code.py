class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def dfs(self, unseen_tree, start, visited=None):
        if visited is None:
            visited = set()
        if start not in self.graph and start in unseen_tree:
            self.graph[start] = unseen_tree[start]

        print(start, end=' ')
        visited.add(start)

        for neighbor in self.graph[start]:
            if neighbor not in visited:
                self.dfs(unseen_tree, neighbor, visited)


if __name__ == "__main__":
    from collections import defaultdict

    """
    输入：一个探索到该节点之前看不见的树
    任务：用深度优先搜索把这棵树搜索一遍
    """

    unseen_tree = {1: [2, 3],
                   2: [4, 5],
                   3: [6]}

    graph = Graph()
    print("DFS traversal starting from node 1:")
    graph.dfs(unseen_tree=unseen_tree, start=1)
