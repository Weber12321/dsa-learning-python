from typing import List


class Solution:
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






if __name__ == '__main__':
    s = Solution()
    lst =[4,5,6,7,0,1,2]
    print(s.search(lst, 5))




