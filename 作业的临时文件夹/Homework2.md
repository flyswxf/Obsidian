# 2.4

![[逆序对示例.png]]
**a.** 
(1,5), (2,5), (3,4), (3,5), (4,5)

**b.** 
逆序对最多的是数组 `n,n - 1,n - 2,...,2,1`。
逆序对数量：$\tfrac{n(n-1)}{2}$。

**c.** 
插入排序(ascending)的计算思路: 
- 外层循环选择元素
- 内层循环寻找插入的位置
在内层循环中, 指针移动的距离就是对当前选择的元素, 前面比它大的数, 也就是以该元素为较小值的逆序对数量. 因为插入排序只改变当前元素之前元素的顺序, 因此每次插入都不会影响后续元素的逆序对.
令逆序对数量为$I$, 则总运行时间$T=\Theta(I)$

**d.**
修改merge-sort的merge过程
算法思路: 在合并两个有序子数组时，若右侧元素 `R[j] < L[i]`，则 `R[j]` 与左侧剩余的所有元素 构成逆序对，数量为 `len(L) - i`。

c++代码：
```cpp
class Solution {
public:
    int cnt=0;
    vector<int> merge(vector<int>& arr1, vector<int>& arr2){
        int n=arr1.size(), m=arr2.size();
        int i=0,j=0;            
        vector<int> res;
        while(i<n&&j<m){
            if(arr1[i]<=arr2[j]){
                res.push_back(arr1[i++]);
            }else{
                res.push_back(arr2[j++]);
                cnt+=n-i;
            }
        }
        while(i<n){
            res.push_back(arr1[i++]);
        }
        while(j<m){
            res.push_back(arr2[j++]);
        }
        return res;
    }
    vector<int> MergeSort(vector<int>& arr, int left, int right){
        int mid=(left+right)>>1;
        vector<int> Left, Right;
        if(left<mid-1){//left...mid-1
            Left=MergeSort(arr, left, mid);
        }else{
            Left.push_back(arr[left]);
        }
        if(mid<right-1){//mid..right-1
            Right=MergeSort(arr,mid,right);
        }else{
            Right.push_back(arr[mid]);
        }
        return merge(Left,Right);
    }
    int reversePairs(vector<int>& record) {
        int n=record.size();
        if(n==0)    return 0;
        MergeSort(record, 0, n);
        return cnt;
    }
};
```

# 6.4.1
Figure 6.4: 
![[堆排序过程分解图.png]]
![[堆排序操作步骤图.png]]
![[堆排序过程图.png]]
- (a): 建堆之后的堆状态
- (b)-(i): 每次调用heapify后的堆状态, i指针指向heapify前堆尾位置, 这个位置将会存放堆首(堆最大值). 黄色节点代表堆外, 蓝色节点代表堆内
- (j) : 堆排序得到的有序数组A

# 6.5.9
![[链表合并示意图.png]]
**参考同类算法**: leetcode: 合并K个升序链表
**模拟输入**：`lists = [[1,4,5],[1,3,4],[2,6]]`
**模拟输出：**`[1,1,2,3,4,4,5,6]`
```cpp
/*
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode() : val(0), next(nullptr) {}
 *     ListNode(int x) : val(x), next(nullptr) {}
 *     ListNode(int x, ListNode *next) : val(x), next(next) {}
 * };
 */
class Solution {
public:
    static void heapify(vector<ListNode*>& a, int i, int n){
        while(true){
            int l=2*i+1;
            int r=l+1;
            int largest=i;
            if(l<n&&a[l]->val<a[largest]->val)    largest=l;
            if(r<n&&a[r]->val<a[largest]->val)    largest=r;
            if(largest==i)   break;
            swap(a[largest],a[i]);
            i=largest;
        }
    }
    static void buildHeap(vector<ListNode*>& arr){
        int n=arr.size();
        for(int i=n/2-1;i>=0;i--){
            heapify(arr,i,n);
        }
    }
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        vector<ListNode*> arr;
        ListNode* dummy=new ListNode();
        ListNode* tail=dummy;
        for(auto& v:lists){
            if(v){ //保证heap中不存在nullptr
                arr.push_back(v);
            }
        }
        int n=arr.size();
        buildHeap(arr);
        while(n>0){
            ListNode* p=arr[0];
            tail->next=p;
            tail=tail->next;
            if(p->next){ //保证heap中不存在nullptr
                arr[0]=p->next;
            }else{
                swap(arr[0],arr[n-1]);
                n--;
            }
            heapify(arr,0,n);
        }
        return dummy->next;
    }
};
```