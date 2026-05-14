import sys


def main():
    # 读取所有输入
    data = sys.stdin.read().split()

    # 如果没有输入数据，直接返回
    if not data:
        return

    # 第一行是测试数据组数T
    t = int(data[0])

    # 存储所有测试用例的n值
    test_cases = []
    for i in range(1, t + 1):
        test_cases.append(int(data[i]))

    # 处理每个测试用例并收集结果
    results = []
    for n in test_cases:
        if n % 2 == 1:
            # 奇数情况，输出-1
            results.append("-1")
        else:
            # 偶数情况，构造排列 [2, 1, 4, 3, 6, 5, ..., n, n-1]
            arr = []
            for i in range(1, n + 1):
                if i % 2 == 1:
                    arr.append(str(i + 1))
                else:
                    arr.append(str(i - 1))
            results.append(" ".join(arr))

    # 按行输出所有结果
    for res in results:
        print(res)


if __name__ == "__main__":
    main()