from collections import defaultdict, Counter
from typing import List

# 69

class Solution:
    # 1. Two Sum
    def twoSum(self, nums, target):
        for idx in range(len(nums)):
            diff = target - nums[idx]
            if diff in nums:
                if idx != nums.index(diff):
                    return [idx, nums.index(diff)]

    # 121: Best Time to Buy and Sell Stock
    def maxProfit(self, prices: List[int]) -> int:
        if len(prices) < 2:
            return 0

        # take the first one as first buying
        purchase = prices[0]
        profit = 0

        for price in prices[1:]:
            # compare previous low-price buying with current price
            purchase = min(purchase, price)
            # compare previous profit with current profit
            profit = max(profit, price-purchase)

        return profit

    # 217: Contains Duplicate
    def containsDuplicate(self, nums: List[int]) -> bool:
        # faster than 56%
        uniq_list = set(nums)
        if len(nums) == len(uniq_list):
            return False
        else:
            return True

    # 238. Product of Array Except Self (X)
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        output = []
        product = 1
        for i in range(len(nums)):
            output.append(product)
            product *= nums[i]
        product = 1
        for i in range(len(nums) -1, -1, -1):
            output[i] *= product
            product *= nums[i]
        return output

    # 53. Maximum Subarray
    def maxSubArray(self, nums: List[int]) -> int:
        if len(nums) == 0:
            return 0
        max_value = nums[0]
        sum_value = nums[0]
        for idx in range(1, len(nums)):
            sum_value = max(sum_value + nums[idx], nums[idx])
            max_value = max(sum_value, max_value)
        return max_value

    # 153. Find Minimum in Rotated Sorted Array
    def findMinimum(self, nums: List[int]) -> int:
        if nums[0] > nums[-1]:
            for i in range(len(nums)-2, -1, -1):
                if nums[i] > nums[i+1]:
                    return nums[i+1]
        else:
            return nums[0]

    # 33. Search in Rotated Sorted Array
    def search(self, nums: List[int], target: int) -> int:
        if target == nums[0]:
            return 0
        if target == nums[-1]:
            return len(nums)-1
        if len(nums) < 2:
            if nums[0] == target:
                return 0
            else:
                return -1
        count = 0
        if target > nums[0]:
            for i in range(len(nums)):
                count+=1
                if nums[i] == target:
                    return i
                elif nums[i] > target:
                    return -1
                else:
                    if count == len(nums):
                        return -1
                    else:
                        continue
        else:
            for i in range(len(nums)-1, -1, -1):
                count += 1
                if nums[i] == target:
                    return i
                elif nums[i] < target:
                    return -1
                else:
                    if count == len(nums):
                        return -1
                    else:
                        continue

    # 15. 3Sum (X)
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        output = []
        i = 0
        while i < len(nums) - 2:
            if nums[i] != nums[i-1] or i == 0:
                target = 0 - nums[i]
                left_pointer = i + 1
                right_pointer = len(nums) - 1
                while left_pointer != right_pointer:
                    if nums[left_pointer] + nums[right_pointer] == target:
                        output.append([nums[i], nums[left_pointer], nums[right_pointer]])
                        # ???
                        while left_pointer < right_pointer:
                            left_pointer += 1
                            if nums[left_pointer] != nums[left_pointer-1]:
                               break
                        while right_pointer > left_pointer:
                            right_pointer -= 1
                            if nums[right_pointer] != nums[right_pointer+1]:
                                break
                    elif nums[left_pointer] + nums[right_pointer] > target:
                        right_pointer -= 1
                    else:
                        left_pointer += 1
            i += 1
        return output

    # 11. Container With Most Water
    def maxArea(self, height: List[int]) -> int:
        max_area= 0
        left_pointer = 0
        right_pointer = len(height) -1

        while left_pointer < right_pointer:
            temp_area = min(height[left_pointer], height[right_pointer])*(right_pointer-left_pointer)
            max_area = max(temp_area, max_area)

            if height[left_pointer] < height[right_pointer]:
                left_pointer += 1
            elif height[left_pointer] > height[right_pointer]:
                right_pointer -= 1
            else:
                left_pointer += 1
                right_pointer -= 1

        return max_area

    # 3. Longest Substring Without Repeating Characters
    def lengthOfLongestSubstring(self, s: str) -> int:
        sub_string = []
        max_len = 0

        for i in range(len(s)):
            if s[i] in sub_string:
                sub_string = sub_string[sub_string.index(s[i])+1:]
            sub_string.append(s[i])
            max_len = max(max_len, len(sub_string))

        return max_len

    # 424. Longest Repeating Character Replacement  (X)
    def characterReplacement(self, s: str, k: int) -> int:
        behind, ahead = 0, 0
        char_count = defaultdict(int)
        char_count[s[ahead]] += 1
        max_length = 0

        while ahead < len(s):
            length = ahead - behind + 1
            max_freq = max(char_count.values())
            if length - max_freq <= k:
                max_length = max(length, max_length)
                ahead += 1
                if ahead == len(s):
                    break
                char_count[s[ahead]] += 1
            else:
                char_count[s[behind]] -= 1
                behind += 1

        return max_length

    # 242. Valid Anagram
    def isAnagram(self, s: str, t: str) -> bool:
        if Counter(s) == Counter(t):
            return True
        else:
            return False

    # 49. Group Anagrams
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        output = defaultdict(list)
        for i in strs:
            sorted_string = ''.join(sorted(i))
            output[sorted_string].append(i)
        return list(output.values())

    # 20. Valid Parentheses
    def isValid(self, s: str) -> bool:
        if len(s) % 2 != 0:
            return False
        dict = {'(': ')', '[': ']', '{': '}'}
        stack = []
        for i in s:
            if i in dict.keys():
                stack.append(i)
            else:
                if stack == []:
                    return False
                a = stack.pop()
                if i != dict[a]:
                    return False
        return stack == []

    # 125. Valid Palindrome
    def isPalindrome(self, s: str) -> bool:
        lst = [char for char in s.lower() if char.isalnum()]
        return lst == lst[::-1]

    # 5. Longest Palindromic Substring (X)
    def longestPalindrome(self, s: str) -> str:
        if len(s) <= 1 or s == s[::-1]:
            return s
        else:
            max_len = 1
            start = 0
            for i in range(1, len(s)):
                odd = s[i - max_len - 1: i + 1]
                even = s[i - max_len: i + 1]
                if i - max_len - 1 >= 0 and odd == odd[::-1]:
                    start = i - max_len - 1
                    max_len = max_len + 2
                    continue
                if even == even[::-1]:
                    start = i - max_len
                    max_len = max_len + 1
        return s[start: start + max_len]

    # 647. Palindromic Substrings (X)
    def countSubstrings(self, s: str) -> int:
        res = 0
        l = len(s)
        for mid in range(l * 2 - 1):
            left = mid // 2
            right = left + mid % 2
            while left >= 0 and right < l and s[left] == s[right]:
                res += 1
                left -= 1
                right += 1
        return res

    # 371. Sum of Two Integers (X)
    def getSum(self, a: int, b: int) -> int:
        # 32 bits integer max
        while b != 0:
            carry = a & b
            a = (a ^ b)
            b = (carry << 1)
        return a if a <= 0x7FFFFFFF else a | (~0x100000000 + 1)

    # 73. Set Matrix Zeroes
    def setZeroes(self, matrix: List[List[int]]) -> None:
        zeros = [0] * len(matrix[0])
        cols = set()
        rows = set()
        for row in range(len(matrix)):
            for col in range(len(matrix[row])):
                if matrix[row][col] == 0:
                    cols.add(col)
                    rows.add(row)

        for row in rows:
            matrix[row] = zeros

        for col in cols:
            for row in range(len(matrix)):
                matrix[row][col] = 0

    # 54. Spiral Matrix
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        output = []
        while matrix:
            output.extend(matrix.pop(0))
            matrix[:] = list(zip(*matrix))[::-1]
        return output

    # 48. Rotate Image
    def rotate(self, matrix: List[List[int]]) -> None:
        # matrix = np.array(matrix)
        # return np.rot90(matrix, 3)
        matrix[::] = list(zip(*matrix[::-1]))

    # 79. Word Search (XX)
    def exist(self, board: List[List[str]], word: str) -> bool:
        seen = set()
        def backtrack(i, j, lenght):
            if lenght == len(word):
                return True
            if i not in range(0, len(board)) or j not in range(0, len(board[0])):
                return False
            cur = (i, j)
            if cur in seen or board[i][j] != word[lenght]:
                return False

            seen.add(cur)
            ret = backtrack(i - 1, j, lenght + 1) or backtrack(i + 1, j, lenght + 1) or backtrack(i, j - 1,
                                                                                                  lenght + 1) or backtrack(
                i, j + 1, lenght + 1)
            seen.remove(cur)
            return ret

        board_letters, word_letters = {}, dict(Counter(word))
        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] not in board_letters:
                    board_letters[board[i][j]] = 1
                else:
                    board_letters[board[i][j]] += 1
        for key in word_letters:
            if key not in board_letters or board_letters[key] < word_letters[key]:
                return False

        for i in range(len(board)):
            for j in range(len(board[0])):
                if backtrack(i, j, 0):
                    return True
        return False

    # 56. Merge Intervals (X)
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if len(intervals) == 0:
            return []
        intervals.sort()
        res = [intervals[0]]
        for interval in intervals[1:]:
            # the next node's smallest value is smaller than the prev node's largest value, then overlapping
            if interval[0] <= res[-1][1]:
                # left boundary is the largest value
                res[-1][1] = max(interval[1], res[-1][1])
            else:
                res.append(interval)
        return res

    # 57. Insert Interval (X)
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        intervals.extend([newInterval])
        intervals.sort()
        results = []
        for si, ei in intervals:
            if not results or results[-1][1] < si:
                results.append([si, ei])
            else:
                results[-1][1] = max(results[-1][1], ei)
        return results

    # 435. Non-overlapping Intervals
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        if len(intervals) == 0:
            return 0
        intervals.sort()
        now = intervals[0][1]
        res = 0
        for i in intervals[1:]:
            if i[0] < now:
                now = min(i[1], now)
                res += 1
            else:
                now = i[1]
        return res

    # 347. Top K Frequent Elements
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = Counter(nums)
        return [c[0] for c in counter.most_common()][0:k]

    # 70. Climbing Stairs
    def climbStairs(self, n: int) -> int:
        # f(n) = f(n-1) + f(n-2)
        lst = [0] * n
        for i in range(n):
            if i == 0:
                lst[i] = 1
            elif i == 1:
                lst[i] = 2
            else:
                lst[i] = lst[i-1] + lst[i-2]
        return lst[n-1]

    # 322. Coin Change (XX)
    def coinChange(self, coins: List[int], amount: int) -> int:
        # dp[i] 表示组成i所需的最少硬币数
        # dp[i]= min(dp[i], dp[i - coin] + 1) for c in coins if i >= coin
        # dp[0] = 0
        # dp[n] 为所求

        dp = [float('inf')] * (amount + 1)
        dp[0] = 0
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] = min(dp[i], dp[i - coin] + 1)
        return dp[amount] if dp[amount] != float('inf') else -1

    # 300. Longest Increasing Subsequence (XX)
    def lengthOfLIS(self, nums: List[int]) -> int:
        if nums == []:
            return 0
        N = len(nums)
        Dp = [1] * N
        for i in range(N - 1):
            for j in range(0, i + 1):
                if nums[i + 1] > nums[j]:
                    Dp[i + 1] = max(Dp[i + 1], Dp[j] + 1)
        return max(Dp)




if __name__ == '__main__':
    s = Solution()
    fn = len([i for i in dir(s) if not i.startswith('__')])
    hard = 3
    print('Number of finished: ', fn)
    print('Number of lasting: ', 75-6-hard-fn)




