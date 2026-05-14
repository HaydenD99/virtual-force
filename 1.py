class Solution:
    def maxProfit(self, prices: List[int], k: int) -> int:
        # write code here
        n = len(prices)
        if n < 2 or k == 0:
            return 0
        max_possible_k = n // 2
        if k > max_possible_k:
            profit = 0
            for i in range(1, n):
                if prices[i] > prices[i - 1]:
                    profit += prices[i] - prices[i - 1]
            return profit

        dp = [[[0] * 2 for _ in range(k + 1)] for i in range(n)]
        for j in range(k + 1):
            dp[0][j][1] = -prices[0]
        for i in range(1, n):
            for j in range(1, k + 1):
                dp[i][j][0] = max(dp[i - 1][j][0], dp[i - 1][j][1] + prices[i])
                dp[i][j][1] = max(dp[i - 1][j][1], dp[i - 1][j - 1][0] - prices[i])

        return max(dp[n - 1][j][0] for j in range(k + 1))