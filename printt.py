def minPathSum(grid):
    if not grid or not grid[0]:
        return 0

    m, n = len(grid), len(grid[0])

    # 初始化dp数组
    dp = [[0] * n for _ in range(m)]

    # 初始化起点
    dp[0][0] = grid[0][0]

    # 初始化第一列
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]

    # 初始化第一行
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]

    # 填充dp数组
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]

    # 返回右下角的值
    return dp[-1][-1]


# 示例1
grid1 = [[1, 3, 1], [1, 5, 1], [4, 2, 1]]
print(minPathSum(grid1))  # 输出: 7

# 示例2
grid2 = [[1, 2, 3], [4, 5, 6]]
print(minPathSum(grid2))  # 输出: 12