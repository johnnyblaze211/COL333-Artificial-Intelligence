import pandas as pd
import numpy as np
import sys

M_val = [1, 0, 0, 0]
A_val = [0, 1, 0, 0]
E_val = [0, 0, 1, 0]
R_val = [0, 0, 0, 1]
'''class Node:
    def __init__(self, array_2D, dict1, pos, pref_order, type1, init_week = False):
        self.array = array_2D
        self.dict = dict1
        self.pos = pos
        self.pref_order = pref_order
        self.type = type1
    


class Tree:
    def __init__(self, N, D, m, a, e, r):
        self.dict = {'N': N, 'D': D, 'm': m, 'a': a, 'e': e, 'r': r}
        if(r > m):
            arr1 = [R_val for i in range(r)] + [M_val for i in range(m)] + [A_val for i in range(a)] + [E_val for i in range(e)]
            arr = np.hstack([np.full((N, D-1), None), np.array(arr1).reshape((-1, 1))])
        pos = (0, D-2)
        if (D-2) % 7 == 6:
            initWeek = True
        self.root = Node(arr, self.dict, pos, pref_order, 1 if r>=m else 0, initWeek = initWeek)
'''
    
def allot_row(N, D, m, a, e, r, prev_row, type1, check_R = True, alloted_R = None):
    new_row = np.full(prev_row.shape, None)
    r_cnt, m_cnt, e_cnt, a_cnt, none_cnt, prev_none_cnt = 0,0,0,0,N,N
    if type1 == 1:
        if not check_R:
            new_row[prev_row == 'M'] = 'R'
            new_row[new_row == None] = ['R']*(r-m) + ['A']*a + ['E']*e + ['M']*(m)
        else:
            new_row[prev_row == 'M'] = 'R'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); r_cnt+= prev_none_cnt-none_cnt
            
            new_row[np.nonzero((new_row == None) & (alloted_R == False))[0][:(r-m)]] = 'R'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); r_cnt+= prev_none_cnt-none_cnt
            
            new_row[np.nonzero((new_row == None) & (alloted_R == False))[0][:m]] = 'M'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); m_cnt+= prev_none_cnt-none_cnt

            new_row[new_row == None] = ['R']*(r-r_cnt) + ['M']*(m-m_cnt) + ['A']*a + ['E']*e
        
    elif type1 == 2:
        if not check_R:
            new_row[np.nonzero((prev_row == 'M') & (new_row == None))[0][:r]] = 'R'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); r_cnt+= prev_none_cnt-none_cnt

            new_row[np.nonzero((prev_row == 'M') & (new_row == None))[0][:a]] = 'A'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); a_cnt+= prev_none_cnt-none_cnt

            new_row[new_row == None] = ['R']*(r - r_cnt) + ['M']*(m - m_cnt) + ['A']*(a - a_cnt) + ['E'] * (e - e_cnt)
        else:

            new_row[np.nonzero((prev_row == 'M') & (alloted_R == False))[0][:r]] = 'R'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); r_cnt+= prev_none_cnt-none_cnt
            
            new_row[np.nonzero((new_row == None) & (prev_row == 'M'))[0][:a]] = 'A'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); a_cnt+= prev_none_cnt-none_cnt
            
            new_row[np.nonzero((prev_row == 'M') & (new_row == None))[0][:(r-r_cnt)]] = 'R'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); r_cnt+= prev_none_cnt-none_cnt

            new_row[np.nonzero((new_row == None) & (alloted_R == False))[0][:(r -r_cnt)]] = 'R'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); r_cnt+= prev_none_cnt-none_cnt

            new_row[np.nonzero((new_row == None) & (alloted_R == False))[0][:r]] = 'M'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); m_cnt+= prev_none_cnt-none_cnt

            new_row[np.nonzero((new_row == None) & (alloted_R == True))[0][:(m-r)]] = 'M'
            prev_none_cnt = none_cnt; none_cnt = (new_row==None).sum(); m_cnt+= prev_none_cnt-none_cnt

            new_row[new_row == None] = ['R']*(r-r_cnt) + ['M']*(m-m_cnt) + ['A']*(a - a_cnt) + ['E']*(e)
        
    return new_row


            



def solution(N, D, m, a, e, r):
    arr = np.full((N, D), None)
    rem = D%7
    init_row = np.array(['R' for i in range(r)] + ['M' for i in range(m)] + ['A' for i in range(a)] + ['E' for i in range(e)])
    alloted_R = (init_row!=init_row)
    prev_row = init_row
    arr[:, -1] = init_row
    if rem == 0:
        for l in range(D-2, -1, -1):
            if l%7 == 6:
                alloted_R = (prev_row!=prev_row)
            else:
                alloted_R = ((prev_row =='R')|(alloted_R == True))

            new_row = allot_row(N, D, m, a, e, r, prev_row, check_R = True, alloted_R = alloted_R, type1 = 1 if r>=m else 2)
            arr[:, l] = new_row
            prev_row = new_row
    else:
        for l in range(D-2, D-rem - 1, -1):
            new_row = allot_row(N, D, m, a, e, r, prev_row, check_R = False, type1 = 1 if r>=m else 2)
            arr[:, l] = new_row
            prev_row = new_row
        for l in range(D-rem-1, -1, -1):
            if l%7 == 6:
                alloted_R = (prev_row!=prev_row)
            else:
                alloted_R = ((prev_row =='R')|(alloted_R == True))

            new_row = allot_row(N, D, m, a, e, r, prev_row, check_R = True, alloted_R = alloted_R, type1 = 1 if r>=m else 2)
            arr[:, l] = new_row
            prev_row = new_row
    

    
    return arr


def check_constraints_a(N, D, m, a, e, r, arr):
    if (arr == None).any(): return False
    for i in range(arr.shape[1]):
        if(arr[:, i] == 'R').sum()!=r: return False
        if(arr[:, i] == 'M').sum()!=m: return False
        if(arr[:, i] == 'A').sum()!=a: return False
        if(arr[:, i] == 'E').sum()!=e: return False
    X = np.argwhere(arr == 'M')
    for i in X:
        if(i[1]>0 and ((arr[i[0], i[1] - 1] == 'E') or (arr[i[0], i[1] - 1] == 'M'))): return False
    weeks = D//7
    for i in range(weeks):
        arrw = arr[:, i*7:(i+1)*7]
        for j in arrw:
            if((j=='R').sum() == 0): return False
    
    return True

            







if __name__ == '__main__':
    params = pd.read_csv('input_a.csv')
    N = int(params['N'])
    D = int(params['D'])
    m = int(params['m'])
    a = int(params['a'])
    e = int(params['e'])
    counter = 0#int(sys.argv[1])
    

    f = open("error_file.txt", "w")
    for N in range(15, 50):
        for D in range(1, 50):
            for m in range(N):
                for a in range(N - m):
                    for e in range(N -m - a):
                        r = N - m - a - e
                        print(f'N: {N}, D: {D}, m: {m}, a: {a}, e: {e}, r: {r}', end = '\r')
                        if(r + a < m):
                            #print()
                            #print('NO-SOLUTION')
                            pass
                        elif(7*r < N and D>=7):
                            #print()
                            #print('NO-SOLUTION')
                            pass
                        else:
                            arr = solution(N, D, m, a, e, r)
                            b1 = check_constraints_a(N, D, m, a, e, r, arr)
                            if(not b1):
                                counter+=1
                                print()
                                print('Returned False')
                                print(arr)
                                f.write(f'Iteration: {counter}\n')
                                f.write(f'N: {N}, D: {D}, m: {m}, a: {a}, e: {e}, r: {r}\n')
                                f.write(arr.__str__())
                                f.write('\n\n')

                                #if not counter: sys.exit()
                            else:
                                #print()
                                #print('YES-SOLUTION')
                                pass
    
    f.close()
    
