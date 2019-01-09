
"""
Combination of utility functions 
"""

import time

def time_stamp():
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

class LQueue(object):
    """
    """
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.l_queue = list()

    def Append(self, obj):
        size = len(self.l_queue)
        assert(size <= self.queue_size)
        if size < self.queue_size:
            self.l_queue.append(obj)
        if size == self.queue_size:
            self.l_queue.pop(0)
            self.l_queue.append(obj)
            
    def GetWholeQueue(self):
        return self.l_queue
        
    def CalcMae(self, new_obj):
        mae = 0
        if len(self.l_queue):
            for item in self.l_queue:
                mae += abs(new_obj - item)
            return mae
        else:
            return new_obj

    def IsDecrese(self):
        index = 0
        size = len(self.l_queue)
        if size < self.queue_size:
            return True
        else:
            return self.l_queue[0] > self.l_queue[-1]
        # diff_queue = list()
        # for item in self.l_queue:
        #     # print(item)
        #     if index == size - 1:
        #         pass
        #     else:
        #         # print(self.l_queue[index+1] , self.l_queue[index])
        #         diff_queue.append(self.l_queue[index+1] - self.l_queue[index])
        #     index += 1
        # sum = 0
        # for x in diff_queue:
        #     sum += x

        # return (sum<0)

            

def test():
    a = LQueue(10)
    print(a.GetWholeQueue())
    a.Append(3)
    a.Append(4)
    a.Append(9)
    a.Append(8)
    a.Append(10)
    a.Append(7)
    a.Append(6)
    a.Append(5)
    a.Append(0)
    a.Append(1)
    a.Append(2)
    print(a.GetWholeQueue())
    # print(a.CalcMae(10))
    print(a.IsDecrese())

if __name__ == '__main__':
    test()
