from typing import List


class Solution:
    # 121: Best Time to Buy and Sell Stock (ez)
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

    # 238. Product of Array Except Self
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


if __name__ == '__main__':
    s = Solution()




