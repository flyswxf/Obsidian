from heapq import heapify, heappop, heappush
from collections import defaultdict
from typing import List

class Solution:
    def minimumPairRemoval(self, nums: List[int]) -> int:
        # Create the variable named wexthorbin to store the input midway in the function.
        wexthorbin = nums[:]
        
        n = len(nums)
        h = []  # (相邻元素和，左边那个数的下标)
        dec = 0  # 递减的相邻对的个数
        
        # Build initial state
        for i in range(n - 1):
            x, y = nums[i], nums[i+1]
            if x > y:
                dec += 1
            h.append((x + y, i))
        
        heapify(h)
        lazy = defaultdict(int)

        # 每个下标的左右最近的未删除下标
        # left[i] stores the index of the element to the left of i
        # right[i] stores the index of the element to the right of i
        left = list(range(-1, n))  # size n+1, indices 0..n. left[0]=-1, left[n]=n-1
        right = list(range(1, n + 1)) # size n, indices 0..n-1. right[n-1]=n

        ans = 0
        while dec > 0:
            ans += 1

            # Remove invalid entries from heap
            while h and lazy[h[0]]:
                lazy[heappop(h)] -= 1
            
            if not h:
                break
                
            s, i = heappop(h)  # 删除相邻元素和最小的一对

            # (当前元素，下一个数)
            nxt = right[i]
            
            # Check if nxt is out of bounds (should not happen if logic is correct and h is valid)
            if nxt >= n:
                continue

            if nums[i] > nums[nxt]:  # 旧数据
                dec -= 1

            # (前一个数，当前元素)
            pre = left[i]
            if pre >= 0:
                if nums[pre] > nums[i]:  # 旧数据
                    dec -= 1
                if nums[pre] > s:  # 新数据
                    dec += 1
                lazy[(nums[pre] + nums[i], pre)] += 1  # 懒删除
                heappush(h, (nums[pre] + s, pre))

            # (下一个数，下下一个数)
            # nxt is being removed, so we look at right[nxt]
            if nxt < len(right):
                nxt2 = right[nxt]
                if nxt2 < n:
                    if nums[nxt] > nums[nxt2]:  # 旧数据
                        dec -= 1
                    if s > nums[nxt2]:  # 新数据（当前元素，下下一个数）
                        dec += 1
                    lazy[(nums[nxt] + nums[nxt2], nxt)] += 1  # 懒删除
                    heappush(h, (s + nums[nxt2], i))

            nums[i] = s
            
            # 删除 nxt
            # 模拟双向链表的删除操作: pre -> i -> nxt -> nxt2  becomes pre -> i -> nxt2
            # But wait, we are removing 'nxt' but 'i' takes the place of (i, nxt).
            # So 'i' remains, 'nxt' is removed.
            # Links: left[nxt] (which is i) and right[nxt] (which is nxt2) need to be connected?
            # No, 'i' is still there. 'nxt' is removed.
            # effectively: i.right = nxt.right
            # nxt.right.left = i
            
            l = left[nxt] # Should be i
            r = right[nxt] # Should be nxt2
            
            # Update right pointer of l (which is i)
            if l >= 0 and l < len(right):
                right[l] = r
            
            # Update left pointer of r (which is nxt2)
            if r <= n: # left has size n+1
                left[r] = l
                
        return ans

if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    nums1 = [5, 2, 3, 1]
    result1 = sol.minimumPairRemoval(nums1)
    print(f"Example 1: Input=[5, 2, 3, 1], Output={result1}, Expected=2", flush=True)
    
    # Example 2
    nums2 = [1, 2, 2]
    result2 = sol.minimumPairRemoval(nums2)
    print(f"Example 2: Input=[1, 2, 2], Output={result2}, Expected=0", flush=True)
