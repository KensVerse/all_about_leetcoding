# inorder traversal
class Solution(object):
    def recoverTree(self, root):
        def inordertraversal(node):
            if not node: return
            inordertraversal(node.left)
            if self.prev and self.prev.val > node.val:
                if not self.first:
                    self.first = self.prev
                self.second = node

            self.prev = node
            inordertraversal(node.right)


        self.first = self.second = self.prev = None
        inordertraversal(root)
        self.first.val, self.second.val = self.second.val, self.first.val

# use one variable to cache entire list's idices
class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        @lru_cache(None)
        def dp(operations, mask):
            if operations == n // 2 + 1: return 0
            res = 0

            for i in range(n - 1):
                if (mask >> i) & 1: continue
                for j in range(i + 1, n):
                    if (mask >> j) & 1: continue
                    newmask = (1 << i) | (1 << j) | mask
                    score = operations * math.gcd(nums[i], nums[j]) + dp(operations + 1, newmask)
                    if score > res: res = score
            return res
        return dp(1, 0)

# variation of Floyd–Warshall
class Solution:
    def calcEquation(self, equations, values, queries):
        dct = collections.defaultdict(dict)
        for (i, j), val in zip(equations, values):
            dct[i][i] = dct[j][j] = 1
            dct[i][j] = val
            dct[j][i] = 1 / val
        for i in dct:
            for j in dct[i]:
                for k in dct[i]:
                    dct[j][k] = dct[j][i] * dct[i][k]
        return [dct[i].get(j, -1) for i, j in queries]

# recursive 1376. Time Needed to Inform All Employees (2023/6/3)
def numOfMinutes(self, n, headID, manager, informTime):
    children = [[] for i in xrange(n)]
    for i, m in enumerate(manager):
        if m >= 0: children[m].append(i)

    def dfs(i):
        return max([dfs(j) for j in children[i]] or [0]) + informTime[i]
    return dfs(headID)

# in order for binary tree, auto sort
L = []
def dfs(node):
    if node.left: dfs(node.left)
    L.append(node.val)
    if node.right: dfs(node.right)
dfs(root)


# use stack to traverse thru the Tree
def maxLevelSum(self, root: TreeNode) -> int:
        ans, q, depth = (-math.inf, 0), [root], -1
        while q:
            ans = max(ans, (sum(node.val for node in q), depth))
            q = [kid for node in q for kid in (node.left, node.right) if kid]
            depth -= 1
        return -ans[1]

# 287. Find the Duplicate Number 6/15/2023, Floyd's algorithm
class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        t, h = nums[0], nums[nums[0]]
        while h != t:
            t = nums[t]
            h = nums[nums[h]]
        t = 0
        while h != t:
            t = nums[t]
            h = nums[h]
        return h
