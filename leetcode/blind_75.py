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

        # faster than 34%
        # d = {}
        # for i in nums:
        #     if i in d:
        #         return True
        #     else:
        #         d[i] = True
        #
        # return False

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


if __name__ == '__main__':
    s = Solution()




