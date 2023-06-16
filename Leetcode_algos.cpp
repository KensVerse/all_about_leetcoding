//6/4/2023 547. Number of Provinces
// UnionFind
class Solution {
public:
    int findCircleNum(vector<vector<int>>& M) {
        if (M.empty()) return 0;
        int n = M.size();

        vector<int> leads(n, 0);
        for (int i = 0; i < n; i++) { leads[i] = i; }   // initialize leads for every kid as themselves

        int groups = n;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {   // avoid recalculate M[i][j], M[j][i]
                if (M[i][j]) {
                    int lead1 = find(i, leads);
                    int lead2 = find(j, leads);
                    if (lead1 != lead2) {       // if 2 group belongs 2 different leads, merge 2 group to 1
                        leads[lead1] = lead2;
                        groups--;
                    }
                }
            }
        }
        return groups;
    }

private:
    int find(int x, vector<int>& parents) {
        return parents[x] == x ? x : find(parents[x], parents);
    }
};

// assymmetric arguments 161. One Edit Distance
//2023/6/4
class Solution {
public:
    bool isOneEditDistance(string s, string t) {
        int m = s.size(), n = t.size();
        if (m > n) {
            return isOneEditDistance(t, s);
        }
        for (int i = 0; i < m; i++) {
            if (s[i] != t[i]) {
                if (m == n) {
                    return s.substr(i + 1) == t.substr(i + 1);
                }
                return s.substr(i) == t.substr(i + 1);
            }
        }
        return m + 1 == n;
    }
};
