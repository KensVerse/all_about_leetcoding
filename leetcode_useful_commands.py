collections.Counter(), collections.Counter(w for w in words if w not in ban).most_common(1)[0][0]
collections.OrderDict() remembers the order of inserted keys; ordereddict.popitem(last=False) gives the first inserted key/value
coll.get("c") # check availability within defaultdict without creating an instance
del cnter1["s"] # delete item "s" from the dict
pc = Counter(filter(lambda x : x.isalpha(), licensePlate.lower()))
return min([w for w in words if Counter(w) & pc == pc], key=len)
x.count(1), x.count("010") # count number of 1's
set([1, 2, 3]).intersection(set([1, 2, 3, 4]))
list(set().union(*d)) # d can be a list of lists
set([1, 2, 3, 4]) - set([1, 2, 3])
set([1, 2, 3, 4]) > set([1, 2, 3]) # includes or not
aset.add(1)
aset.remove(1)
list((Counter([1, 2, 3, 4, 5, 5]) - Counter([1, 2, 3])).elements())
max([dfs(j) for j in children[i]] or [0]) # incase the first list is empty
binary search: l = mid + 1, cannot let l=mid-> infinite loop if l + 1 = r and l=mid's condition is met

ranges += [], # add a tuple containing a "[]" to the list
ranges[-1][1:] = n, # assign a tuple containing n to the last item in ranges, if the last item is empty, it assignes to n to be its only value
d = collections.deque([(1, 2, 3)]) # fast popleft(), as list's pop(0) is of N complexity due to the shifting of entire list
deque usually applied to a list as argument
collections.defaultdict(list)
x = collections.defaultdict(lambda: 0)
x.pop(certainkey)
x[i, j] += 1
dict1.update(dict2) # add s1 and replace s1 with s2
"and" has higher priority than "or": try 0 and 0 or 1
sorted(counts, key=counts.get, reverse=True)
function within a function: res += xxx wouldn't work, using self.res += xxxx, as "res" is an iteger (reference before assignment error)

x = [1]
for i in x: # loop is extendable
    print(i)
    if i + 10 < 15:
        x.append(i + 1)

b = [1]
for i in b:
    for j in range(i + 1, i + 7):
        row, col = (j - 1) // n, (j - 1) % n
        val = board[-(row + 1)][col if row % 2 == 0 else -(col + 1)] # bitwise not (~) is equal to -(x + 1) = ~x
        if j not in seen:
            b.append(j) # append during a loop, iterate over the appended values

for i, j, k in [[1, 2, 3], [3, 4, 5]]
alist.extend(anotherlist_or_a_tuple)
max(0.5 * abs((j[0] - i[0]) * (k[1] - i[1]) - (j[1] - i[1]) * (k[0] - i[0])) for i, j, k in itertools.combinations(points, 3))
list(itertools.accumulate([1, 2, 3]))
accumulate([0, 7, 19, 13], lambda a, b: b - a)
pow(2, (r - l), mod) # faster than 2 ** x % mod

def first(l, h, check): # pass lambda as an argument
    while l < h:
        m = (l + h) // 2
        if check(m):
            h = m
        else:
            l = m + 1
    return l
top = first(0, x, lambda x: "1" in image[x])

[[(ch, len(list(g)))] for ch, g in itertools.groupby("vtvkgn")]# groupby is an iterator, so it changes after every time it's called
all(list comprehension) # might be evaluated simultaneous, not one after each other, if the first item's function changes subsequent values, all() doesn't take the changes into consideration
['yes' if v == 1 else 'no' if v == 2 else 'idle' for v in l]
[[ab_dict[j] for j in i] for i in words] # list of list
uniq = []
[uniq.append(num) for num in nums if nums.count(num) == 1]
def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        memo = {len(s): ['']}
        def sentences(i):
            if i not in memo:
                memo[i] = [s[i:j] + (tail and ' ' + tail) #if tail not "", then add space " " and then tail, if "", add nothing
                        for j in range(i+1, len(s)+1)
                        if s[i:j] in wordDict
                        for tail in sentences(j)]
            return memo[i]
        return sentences(0)

no: [[0] * len(mat)] * len(mat) # if i change one of the lists inside, all other lists change in the same way
yes: [[0] * len(mat) for _ in range(len(mat))] # each list component changes independently

data = iter(data.split()) # generator
val = next(vals)
random.choices(population, weights, k=1) # random.choices(nums) select an item based on probability
idx = random.randint(0, len(nums) - 1)

# area for parallelogram direction (2, 3) and (4, 1) is abs(2 * 1 - 3 * 4) cross product
import functools
def gcd(a, b): # greatest common numbers, or use math.gcd; least common multiple use math.
    while b:
        a, b = b, a % b
    return a
functools.reduce(gcd, [6, 12, 15, 21])
import copy
x = copy.copy(x)

words = re.findall(r'\w+', paragraph.lower())
s.replace("banana", "apple").replace("a", "b")
' '.join(latin(w, i) for i, w in enumerate(S.split()))
[(a, b) for a, b in zip(s, goal) if a != b]
itertools.zip_longest('GesoGes', 'ekfrek', fillvalue ='_' )
sum(map(max, zip(*grid)))
map(lambda x: x ** 2, s_list)
c = lambda: {(x, y), (x, Y), (X, y), (X, Y)}
for x, y, X, Y in rectangles: # changing values of arguments
    corners ^= c()
 (f(z) for f, z in zip((min, min, max, max), zip(*rectangles))) # function is list comprehension
X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]
Z = [x for _,x in sorted(zip(Y,X))]
x.sort(key=lambda x: x[1])
x.reverse()
max(part1, part2, key=len)

heap = [] #heapq not necessarily gives a sorted list, but always pop the minimum number
import heapq
heapq.heappush(heap, 2)
heapq.heappush(heap, 1)
heapq.heappush(heap, 3)
heapq.heappop(heap)
or
x = [1, 3, 2]
heapq.heapify(x)
heapq.heappop(x) # return the smallest elementa\
heapq.heapreplace(heap, i) # pop the smallest and push the new one
heappushpop(self.pq, nums[i]) # push an item and pop the smallest one

def alien_compare(x, y):
    for i, j in zip(x, y):
        if i != j:
            return order.index(i) - order.index(j)
    return -1 if len(x) < len(y) else 1
 words == sorted(words, cmp=alien_compare)

 sorted(words, key=lambda w: map(order.index, w))
 sorted(arr, key=lambda x: (bin(x).count('1'), x)) # 2 keys


import bisect
bisect.bisect_left([1, 2, 3, 4, 4, 7, 8, 9], 6) #obtain loc of 6
bisect.insort(prefix_sums, s_curr) # insert value in order
set("abc") & set("bcd") # overlapping and union of sets
xor operator: x ^ y
binary and: x & (-x) greatest 2**k divisor
list(filter(selected.isdisjoint, less)) # filter elements in less (a list) for elements which don't contain elements in the set "selected"
valid = filter(xx, xx)
apply list(valid) twice in a row, the second time will return an empty list as "filter" is an iterator

x.isalpha() # check whether it's alphabet
x.isdigit()
alphabet = list(string.ascii_lowercase) # list from a to z
x.upper()
x.swapcase()
re.search('[a-z]', password)

min([indices[i][-1] - indices[i][0] + 1 for i in cnter if cnter[i] == max_cnter])

x = [1, 2, 3]
x.rindex(2) # the rightmost occurance's index
x.remove(3) applies to set too # based on Value
del x[-1] # based on index
x.pop() or pop(0), pop(3) #delete the 3rd element
x.insert(0, -1)
[1, 2, 3] == [1, 2, 4]
try:
    j = arr.index(arr[i]+d, i+1)
    k = arr.index(arr[j]+d, j+1)
except ValueError:
    pass
x = "abca"
x.strip("a") # get rid of "a" at the beginning and at the end

#
# break out of outer loop: for else, the part below "else" only execute when the inner loop doesn't break
res = 1
for i in range(3, n + 1):
    if i % 2 == 0:
        continue
    sqr = int(i ** 0.5)
    for j in range(3, sqr + 1):
        if i % j == 0:
            break
    else:
        res += 1

# use itself as a function
class Solution:
def isScramble(self, s1, s2):
    f = self.isScramble
    for i in range(1, n):
        if f(s1[:i], s2[:i]) and f(s1[i:], s2[i:]) or \
           f(s1[:i], s2[-i:]) and f(s1[i:], s2[:-i]):
            return True
    return False

# loop using set and "seen"
seen = set()
res = float('inf') # or inf
for x1, y1 in points:
    for x2, y2 in seen:
        if (x1, y2) in seen and (x2, y1) in seen:
            area = abs(x1 - x2) * abs(y1 - y2)
            if area and area < res:
                res = area
    seen.add((x1, y1))




# Hackerrank unusable
from sortedcontainers import SortedList # sorted list uses add instead of append

# Hackerrank specific
if (x.next != None) # instead of "if x.next"
use "is None" as a general rule
# generator
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        for c1, c2 in zip(self.generate(word1), self.generate(word2)):
            if c1 != c2:
                return False
        return True

    def generate(self, wordlist: List[str]):
        for word in wordlist:
            for char in word:
                yield char
        yield None
# memorization
class Solution:
    @lru_cache(None)
    def encode(self, s: str) -> str:
        i=(s+s).find(s,1)
        encoded=str(len(s)//i)+'['+self.encode(s[:i])+']' if i<len(s) else s
        splitEncoded=[self.encode(s[:i])+self.encode(s[i:]) for i in range(1,len(s))]
        return min(splitEncoded+[encoded],key=len)
# depth first search
#
# breadth first search

# Trees
def preOrder(root):
    #Write your code here
    print(root.info, end=' ')
    if root.left:
        preOrder(root.left)
    if root.right:
        preOrder(root.right)
# https://leetcode.com/problems/construct-string-from-binary-tree/submissions/
class Solution:
    def tree2str(self, root: Optional[TreeNode]) -> str:
        res = str(root.val)
        if root.left:
            res += "(" + self.tree2str(root.left) + ")"
        if root.right:
            if not root.left:
                res += "()" + "(" + self.tree2str(root.right) + ")"
            else:
                res += "(" + self.tree2str(root.right) + ")"
        return res
#
def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
	arr = sorted(set(self.treeToList(root)))
	if len(arr) < 2:
		return -1
	else:
		return arr[1]

def treeToList(self, root):
	if root is None:
		return []
	return self.treeToList(root.left) + [root.val] + self.treeToList(root.right)
#
class Solution(object):
    def findSecondMinimumValue(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        collected = [root]
        res = []
        while collected:
            current_node = collected.pop(0)
            res.append(current_node.val)
            if current_node.left:
                collected.append(current_node.left)
            if current_node.right:
                collected.append(current_node.right)
        res = sorted(list(set(res)))
        if len(res) > 1:
            return res[1]
        else:
            return -1

class Solution(object):
    def floodFill(self, image, sr, sc, color):
        """
        :type image: List[List[int]]
        :type sr: int
        :type sc: int
        :type color: int
        :rtype: List[List[int]]
        """
        start_color = image[sr][sc]

        def fill(sr, sc):
            if sr < 0 or sr > len(image) - 1: return
            if sc < 0 or sc > len(image[0]) - 1: return

            if image[sr][sc] == color: return
            if image[sr][sc] != start_color: return
            image[sr][sc] = color

            fill(sr + 1, sc)
            fill(sr - 1, sc)
            fill (sr, sc + 1)
            fill(sr, sc - 1)

        fill(sr, sc)
        return image

def reverse(head):
    while head.next!=None:

        #swap the pointer
        head.next,head.prev,head=head.prev,head.next,head.next

    #change to the tail node
    head.next,head.prev=head.prev,None
    return head
