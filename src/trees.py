import time


class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None


class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if not self.root:
            self.root = BSTNode(key)
            return
        cur = self.root
        while True:
            if key == cur.key:
                return
            if key < cur.key:
                if cur.left:
                    cur = cur.left
                else:
                    cur.left = BSTNode(key)
                    return
            else:
                if cur.right:
                    cur = cur.right
                else:
                    cur.right = BSTNode(key)
                    return

    def search(self, key):
        cur = self.root
        while cur:
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False


class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.keys = []
        self.children = []
        self.leaf = leaf


class BTree:
    def __init__(self, t=3):
        self.root = BTreeNode(t, leaf=True)
        self.t = t

    def search(self, k, x=None):
        if x is None:
            x = self.root
        i = 0
        while i < len(x.keys) and k > x.keys[i]:
            i += 1
        if i < len(x.keys) and k == x.keys[i]:
            return True
        if x.leaf:
            return False
        return self.search(k, x.children[i])

    def insert(self, k):
        r = self.root
        if len(r.keys) == (2*self.t - 1):
            s = BTreeNode(self.t, leaf=False)
            s.children.insert(0, r)
            self._split_child(s, 0)
            self._insert_nonfull(s, k)
            self.root = s
        else:
            self._insert_nonfull(r, k)

    def _split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = BTreeNode(t, leaf=y.leaf)
        z.keys = y.keys[t:]
        y.keys = y.keys[:t-1]
        if not y.leaf:
            z.children = y.children[t:]
            y.children = y.children[:t]
        x.children.insert(i+1, z)
        x.keys.insert(i, y.keys.pop())

    def _insert_nonfull(self, x, k):
        if x.leaf:
            i = len(x.keys)-1
            x.keys.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i+1] = x.keys[i]
                i -= 1
            x.keys[i+1] = k
        else:
            i = len(x.keys)-1
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == 2*self.t - 1:
                self._split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self._insert_nonfull(x.children[i], k)

def time_search_structure(struct, queries):
    import time
    start = time.perf_counter()
    for q in queries:
        struct.search(q)
    end = time.perf_counter()
    return end-start
