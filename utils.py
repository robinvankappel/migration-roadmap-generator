# Backtracking based Python3 program to prall
# permutations of a string that follow given
# constraint
import numpy as np
import time
import settings


def filter_where(arr, k):
    return arr[np.where(arr[0] == k)]


def isSafe(lst, condition):
    # If first element is not found in inclusion list
    # then do not proceed.
    if 0 in condition:
        return False
    for c in condition:
        if lst[0] == c:
            return True
    # if (lst[0] == condition) == False:
    #     return False
    # elif sum(lst[0] == condition):
    #     return True
    return False


def permute(lst, l, r, condition=[0]):
    # We reach here only when permutation
    # is valid
    if (l == r):
        # print(*lst, sep="", end=" ")
        return
    # Fix all elements one by one
    for i in range(l, r + 1):
        # Fix lst[i] only if it is a valid move.
        lst[l], lst[i] = lst[i], lst[l]
        if (isSafe(lst, condition)):
            print(lst)
            permute(lst, l + 1, r)
        # print(lst)
        # lst[l], lst[i] = lst[i], lst[l]


def permute_heap(l, startvalue=None):
    n = len(l)
    result = []
    c = n * [0]

    result.append(l)

    i = 0
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                tmp = l[0]
                l[0] = l[i]
                l[i] = tmp

            else:

                tmp = l[c[i]]
                l[c[i]] = l[i]
                l[i] = tmp

            c[i] += 1
            i = 0

            # if condition for when to append:
            if not startvalue == None:
                if c[-1] == (len(startvalue)):
                    break
                if any(l[-1] == s for s in startvalue):
                    # todo: WARNING currently not working for multiple start values, only single value
                    # todo: wrong assumption that next iteration starts with next startvalue
                    # todo: right way would be to generate all permutations of the startvalues and add them to each permutation of the reduced list.
                    l_reverse = l[::-1]
                    result.append(l_reverse)
                    if len(result) % 1000000 == 0:
                        print('# roadmaps: ', len(result) / 1000000, 'M')
        else:
            c[i] = 0
            i += 1
    return result


# Original Heap permutation function: https://stackoverflow.com/questions/29042819/heaps-algorithm-permutation-generator
def heap_perm_(n, A):  # n = len(A), A = permutation list
    if n == 1:
        yield A
    else:
        for i in range(n - 1):
            for hp in heap_perm_(n - 1, A): yield hp
            j = 0 if (n % 2) == 1 else i
            A[j], A[n - 1] = A[n - 1], A[j]
        for hp in heap_perm_(n - 1, A): yield hp


def heap(a):
    """
    This program will take any iterable object
    as input and will give a generator object
    as output which can be used with for
    loop to get all the permutations.
    To print: use list(a) or to print a part use [next(a) for i in range(X)]
    """
    n = len(a)
    c = [0] * n
    A = list(a)
    yield A
    i = 1
    while i < n:
        if c[i] < i:
            if i % 2 == 0:
                temp = A[0]
                A[0] = A[i]
                A[i] = temp
            else:
                temp = A[c[i]]
                A[c[i]] = A[i]
                A[i] = temp
            yield A
            c[i] += 1
            i = 1
        else:
            c[i] = 0
            i += 1

def get_time(t):
    """
    Time difference between now and t
    """
    time_diff = round(time.time() - t,5)
    settings.time0 = time.time()
    return time_diff