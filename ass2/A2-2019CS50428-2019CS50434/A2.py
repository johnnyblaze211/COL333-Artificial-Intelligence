import numpy as np
import sys
import pandas as pd
import json
import time
    
calls = 0
best_val = -1
best_arr = None
start_time = None
print_counter = 0
div_max_time = None
def numpy_to_dict(arr):
    N = arr.shape[0]
    D = arr.shape[1]
    dict1 = {}
    for i in range(N):
        for j in range(D):
            dict1[f'N{i}_{j}'] = arr[i][j]
    return dict1
class Tree:
    def __init__(self, N, D, m, a, e, arr, leftmostRow, alloted_R, part = 'a'):
        self.N = N
        self.D = D
        self.arr = arr
        self.m = m
        self.a = a
        self.e = e
        self.r = N - m - a - e
        self.leftmostRow = leftmostRow
        self.alloted_R = alloted_R
        self.curr_row = arr[:, leftmostRow]
        self.idxs_M_R = (self.curr_row == 'M') & (alloted_R == True)
        self.idxs_M = (self.curr_row == 'M') & (alloted_R  == False)
        self.idxs_R = (self.curr_row != 'M') & (alloted_R == True)
        self.idxs_ = (self.curr_row != 'M') & (alloted_R == False)

    def expand(self, debug = False):
        global calls
        calls+=1
        sum_M_R = self.idxs_M_R.sum()
        sum_M = self.idxs_M.sum()
        sum_R = self.idxs_R.sum()
        sum_ = self.idxs_.sum()

        a = self.a; m = self.m; e = self.e; r = self.r; N = self.N;
        a_cnt = a; m_cnt = m; e_cnt = e; r_cnt = r;
        if debug:
            print('Expanded')
            print(self.curr_row)
            print(np.nonzero(self.idxs_M_R)[0])
            print(np.nonzero(self.idxs_M)[0])
            print(np.nonzero(self.idxs_R)[0])
            print(np.nonzero(self.idxs_)[0])
            print(self.arr)
            print(f'leftmostRow: {self.leftmostRow}')
            print('\n\n\n\n')
        for r_M in range(min(sum_M, r_cnt), max(sum_M - a_cnt, 0) - 1, -1):
            if debug: print(f'{min(sum_M, r_cnt)}>=r_M>={max(sum_M - a_cnt, 0)}')
            a_M = sum_M - r_M
            a_cnt1 = a_cnt - a_M
            r_cnt1 = r_cnt - r_M
            #print(f'r_cnt:{r_cnt1}')
            if debug: print(f'current r_M, a_M, s_M: {r_M, a_M, sum_M}')

            for r_M_R in range(max(sum_M_R - a_cnt1, 0), min(sum_M_R, r_cnt1) + 1, 1):
                if debug: print(f'{max(sum_M_R - a_cnt1, 0)}<=r_M_R<={min(sum_M_R, r_cnt1)}')
                a_M_R = sum_M_R - r_M_R
                a_cnt2 = a_cnt1 - a_M_R
                r_cnt2 =r_cnt1 - r_M_R
                if debug: print(f'current r_M_R, a_M_R, s_M_R: {r_M_R, a_M_R, sum_M_R}')

                for r_ in range(min(sum_, r_cnt2), max(0, sum_ - m - a_cnt2 - e) -1, -1):


                    if debug: print(f'{min(sum_,r_cnt2)}>=r_>={max(0, sum_ - m - a_cnt2 - e)}')
                    if debug: print(f'current r_: {r_}')
                    for m_ in range(min(sum_ - r_, m_cnt), max(0, sum_ - r_ - a_cnt2 - e) - 1, -1):
                        a_ = min(sum_ - r_ - m_, a_cnt2)
                        e_ = sum_ - r_ - m_ - a_
                        a_cnt3 = a_cnt2 - a_
                        m_cnt3 = m_cnt - m_
                        e_cnt3 = e_cnt - e_
                        r_cnt3 = r_cnt2 - r_

                        
                        if debug: print(f'{min(sum_ - r_, m_cnt)}>=m_>={max(0, sum_ - r_ - a - e)}')
                        if debug: print(f'current r_, m_, a_, e_, s_: {r_, m_, a_, e_, sum_}')


                        a_R = a - a_M - a_M_R - a_; assert(a_R == a_cnt3)
                        m_R = m - m_; assert(m_R == m_cnt3)
                        e_R = e - e_; assert(e_R == e_cnt3)
                        r_R = r - r_M -r_M_R - r_; assert(r_R == r_cnt3)

                        if debug: print(f'current r_R, m_R, a_R, e_R, sum_R: {r_R, m_R, a_R, e_R, sum_R}')
                        new_row = np.full(self.curr_row.shape, None)
                        new_row[np.nonzero(self.idxs_M)[0]] = ['R']*(r_M) + ['A']*(a_M)
                        new_row[np.nonzero(self.idxs_M_R)[0]] = ['R']*(r_M_R) + ['A'] * (a_M_R)
                        new_row[np.nonzero(self.idxs_)[0]] = ['R']*r_ + ['A']*a_ + ['M']*m_ + ['E']*e_
                        new_row[np.nonzero(self.idxs_R)[0]] = ['R']*r_R + ['A']*a_R + ['M']*m_R + ['E']*e_R
                        '''try:
                            new_row[np.nonzero(self.idxs_R)[0]] = ['R']*r_R + ['A']*a_R + ['M']*m_R + ['E']*e_R
                        except:
                            print('yoo')
                            print(['R']*r_R + ['A']*a_R + ['M']*m_R + ['E']*e_R)
                            print(self.curr_row)
                            print(np.nonzero(self.idxs_M_R)[0])
                            print(np.nonzero(self.idxs_M)[0])
                            print(np.nonzero(self.idxs_R)[0])
                            print(np.nonzero(self.idxs_)[0])
                            print(self.arr)
                            
                            sys.exit()
                        '''
                        
                        assert((new_row == None).sum() == 0)
                        new_alloted_R = (new_row == 'R') | (self.alloted_R)
                        if (new_alloted_R == False).sum() > r*(self.leftmostRow - 1): continue
                        if self.leftmostRow == 2:
                            min_r_fixed = max(m-a, 0)
                            s1 = (new_row[new_alloted_R == False] != 'M').sum()
                            if r - min_r_fixed < s1: continue
                        newarr = self.arr.copy()
                        newarr[:, self.leftmostRow - 1] = new_row
                        if self.leftmostRow == 1: return True, newarr
                        else:
                            newtree = Tree(self.N, self.D, m, a, e, newarr, self.leftmostRow - 1, new_alloted_R)
                            if debug: print('start')
                            res = newtree.expand(debug=debug)
                            if debug: print(res)
                            
                            if res[0]:
                                return True, res[1]
                            else: 
                                if debug: print('yay')
                                continue

        
        return False, None

def check_constraints_a(N, D, m, a, e, r, arr):
    assert(arr.shape[0] == N)
    assert(arr.shape[1] == D)
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

def check_constraints_b(N, D, m, a, e, r, S, arr):
    try:
        if(check_constraints_a(N, D, m, a, e, r, arr)):
            val = ((arr[:S] == 'M')|(arr[:S] == 'E')).sum()
            return True, val
        return False, None


    except AssertionError:
        return False, None


class Tree2:
    def __init__(self, N, D, m, a, e, S, arr, leftmostRow, alloted_R, count_arr):
        self.N = N
        self.D = D
        self.m = m
        self.e = e
        self.a = a
        self.S = S
        self.arr = arr
        self.leftmostRow = leftmostRow
        self.alloted_R = alloted_R
        self.count_arr = count_arr
        self.curr_row = arr[:, leftmostRow]
        S_mask = np.full(self.curr_row.shape, False)
        S_mask[:S] = True
        self.S_mask = S_mask
        self.idxs_M_R_S = (self.curr_row == 'M') & (alloted_R == True) & (self.S_mask)
        self.idxs_M_S = (self.curr_row == 'M') & (alloted_R  == False) & (self.S_mask)
        self.idxs_R_S = (self.curr_row != 'M') & (alloted_R == True) & (self.S_mask)
        self.idxs_S = (self.curr_row != 'M') & (alloted_R == False) & (self.S_mask)

        self.idxs_M_R = (self.curr_row == 'M') & (alloted_R == True) & (self.S_mask == False)
        self.idxs_M = (self.curr_row == 'M') & (alloted_R  == False) & (self.S_mask == False)
        self.idxs_R = (self.curr_row != 'M') & (alloted_R == True) & (self.S_mask == False)
        self.idxs_ = (self.curr_row != 'M') & (alloted_R == False) & (self.S_mask == False)

    def expand(self, debug = False):
        global best_arr, best_val, calls, start_time, print_counter
        calls+=1

        sum_M_R_S = self.idxs_M_R_S.sum()
        sum_M_S = self.idxs_M_S.sum()
        sum_R_S = self.idxs_R_S.sum()
        sum_S = self.idxs_S.sum()

        sum_M_R = self.idxs_M_R.sum()
        sum_M = self.idxs_M.sum()
        sum_R = self.idxs_R.sum()
        sum_ = self.idxs_.sum()
        a = self.a; m = self.m; e = self.e; N = self.N; r = N-m-a-e
        a_cnt = a; m_cnt = m; e_cnt = e; r_cnt = r;

        if debug:
            print('Expanded')
            print(self.curr_row)
            print(f'idxs_M_R_S{np.nonzero(self.idxs_M_R_S)[0]}')
            print(f'idxs_M_S{np.nonzero(self.idxs_M_S)[0]}')
            print(f'idxs_R_S{np.nonzero(self.idxs_R_S)[0]}')
            print(f'idxs_S{np.nonzero(self.idxs_S)[0]}')
            print(f'idxs_M_R{np.nonzero(self.idxs_M_R)[0]}')
            print(f'idxs_M{np.nonzero(self.idxs_M)[0]}')
            print(f'idxs_R{np.nonzero(self.idxs_R)[0]}')
            print(f'idxs_{np.nonzero(self.idxs_)[0]}')
            print(self.arr)
            print(f'leftmostRow: {self.leftmostRow}')
            print('\n\n\n\n')

        bool_break_loops = False
        for r_M in range(min(sum_M, r_cnt), max(sum_M - a_cnt, 0) - 1, -1):
            if bool_break_loops: break
            #if debug: print(f'{min(sum_M, r_cnt)}>=r_M>={max(sum_M - a_cnt, 0)}')
            a_M = sum_M - r_M
            a_cnt1 = a_cnt - a_M
            r_cnt1 = r_cnt - r_M
            m_cnt1 = m
            e_cnt1 = e
            #if debug: print(f'current r_M, a_M, s_M: {r_M, a_M, sum_M}')

            for r_M_R_S in range(max(sum_M_R_S - a_cnt1, 0), min(sum_M_R_S, r_cnt1)+1):
                if bool_break_loops: break
                #if debug: print(f'{max(sum_M_R_S - a_cnt1, 0)}<=r_M<={min(sum_M_R_S, r_cnt1)}')
                a_M_R_S = sum_M_R_S - r_M_R_S
                a_cnt2 = a_cnt1 - a_M_R_S
                r_cnt2 = r_cnt1 - r_M_R_S
                m_cnt2 = m_cnt1
                e_cnt2 = e_cnt1
                #if debug: print(f'current r_M_R_S, a_M_R_S, s_M_R_S: {r_M_R_S, a_M_R_S, sum_M_R_S}')

                for r_M_R in range(max(sum_M_R - a_cnt2, 0), min(sum_M_R, r_cnt2) + 1):
                    if bool_break_loops: break
                    #if debug: print(f'{max(sum_M_R - a_cnt2, 0)}<=r_M_R<={min(sum_M_R, r_cnt2)}')
                    a_M_R = sum_M_R - r_M_R
                    a_cnt3 = a_cnt2 - a_M_R
                    r_cnt3 = r_cnt2 - r_M_R
                    m_cnt3 = m_cnt2
                    e_cnt3 = e_cnt2
                    #if debug: print(f'current r_M_R, a_M_R, s_M_R: {r_M_R, a_M_R, sum_M_R}')
                    #pass

                    for r_M_S in range(min(sum_M_S, r_cnt3), max(sum_M_S - a_cnt3, 0) - 1, -1):
                        if bool_break_loops: break
                        #if debug: print(f'{min(sum_M_S, r_cnt3)}>=r_M>={max(sum_M_S - a_cnt3, 0)}')
                        a_M_S = sum_M_S - r_M_S
                        a_cnt4 = a_cnt3 - a_M_S
                        r_cnt4 = r_cnt3 - r_M_S
                        m_cnt4 = m_cnt3
                        e_cnt4 = e_cnt3
                        #if debug: print(f'current r_M_S, a_M_S, s_M_S: {r_M_S, a_M_S, sum_M_S}')
                        #pass
                

                



                        for r_ in range(min(sum_, r_cnt4), max(sum_ - a_cnt4 - e_cnt4 - m_cnt4, 0) -1, -1):
                            if bool_break_loops: break
                            for e_ in range(max(sum_ - a_cnt4 - r_ - m_cnt4, 0), min(sum_ - r_, e_cnt4) + 1):
                                if bool_break_loops: break
                                for m_ in range(max(sum_ - a_cnt4 - r_ - e_, 0), min(sum_ - r_ - e_, m_cnt4)+1):
                                    if bool_break_loops: break
                                    a_ = sum_ - r_ - m_ - e_
                                    a_cnt5 = a_cnt4 - a_
                                    m_cnt5 = m_cnt4 - m_
                                    e_cnt5 = e_cnt4 - e_
                                    r_cnt5 = r_cnt4 - r_

                                    for r_R_S in range(max(sum_R_S - a_cnt5 - m_cnt5 - e_cnt5, 0), min(sum_R_S, r_cnt5)+1):
                                        if bool_break_loops: break
                                        for e_R_S in range(min(sum_R_S - r_R_S, e_cnt5), max(sum_R_S - a_cnt5 - m_cnt5 - r_R_S, 0)-1, -1):
                                            if bool_break_loops: break
                                            for m_R_S in range(min(sum_R_S - r_R_S - e_R_S, m_cnt5), max(sum_R_S - a_cnt5 - e_R_S - r_R_S, 0)-1, -1):
                                                if bool_break_loops: break
                                                a_R_S = sum_R_S - r_R_S - e_R_S - m_R_S
                                                a_cnt6 = a_cnt5 - a_R_S
                                                m_cnt6 = m_cnt5 - m_R_S
                                                e_cnt6 = e_cnt5 - e_R_S
                                                r_cnt6 = r_cnt5 - r_R_S

                                                for r_S in range(min(sum_S, r_cnt6), max(sum_S - a_cnt6 - e_cnt6 - m_cnt6, 0)-1, -1):
                                                    if bool_break_loops: break
                                                    for e_S in range(min(sum_S - r_S, e_cnt6), max(sum_S - a_cnt6 - m_cnt6 - r_S, 0) - 1, -1):
                                                        if bool_break_loops: break
                                                        for m_S in range(min(sum_S - r_S - e_S, m_cnt6), max(sum_S - a_cnt6 - e_S - r_S, 0) - 1, -1):
                                                            if bool_break_loops: break
                                                            if((time.time() - start_time) > print_counter*div_max_time/10):
                                                                print(f'Recursive calls so far: {calls}')
                                                                #print(best_val)
                                                                #print(best_arr)
                                                                print_counter+=1
                                                            if(time.time() - start_time > div_max_time): 
                                                                return None
                                                            a_S = sum_S - r_S - e_S - m_S
                                                            a_cnt7 = a_cnt6 - a_S
                                                            e_cnt7 = e_cnt6 - e_S
                                                            m_cnt7 = m_cnt6 - m_S
                                                            r_cnt7 = r_cnt6 - r_S

                                                            a_R = a - (a_M_R_S + a_M_S + a_R_S + a_S + a_M_R + a_M + a_)
                                                            e_R = e - (e_R_S + e_S + e_)
                                                            m_R = m - (m_R_S + m_S + m_)
                                                            r_R = r - (r_M_R_S + r_M_S + r_R_S + r_S + r_M_R + r_M + r_)

                                                            assert(a_R == a_cnt7)
                                                            assert(e_R == e_cnt7)
                                                            assert(m_R == m_cnt7)
                                                            assert(r_R == r_cnt7)

                                                            new_row = np.full(self.curr_row.shape, None)
                                                            if self.leftmostRow%7 != 0:
                                                                new_row[np.nonzero(self.idxs_)[0]] = ['R']*r_ + ['M']*m_ + ['E']*e_ + ['A']*a_
                                                                new_row[np.nonzero(self.idxs_M)[0]] = ['R']*r_M + ['A']*a_M
                                                                new_row[np.nonzero(self.idxs_R)[0]] = ['R']*r_R + ['M']*m_R + ['E']*e_R + ['A']*a_R
                                                                new_row[np.nonzero(self.idxs_S)[0]] = ['R']*r_S + ['M']*m_S + ['E']*e_S + ['A']*a_S
                                                                new_row[np.nonzero(self.idxs_R_S)[0]] = ['R']*r_R_S + ['M']*m_R_S + ['E']*e_R_S + ['A']*a_R_S
                                                                new_row[np.nonzero(self.idxs_M_R)[0]] = ['R']*r_M_R + ['A']*a_M_R
                                                                new_row[np.nonzero(self.idxs_M_S)[0]] = ['R']*r_M_S + ['A']*a_M_S
                                                                new_row[np.nonzero(self.idxs_M_R_S)[0]] = ['R']*r_M_R_S + ['A']*a_M_R_S


                                                                
                                                                

                                                            assert((new_row == None).sum() == 0)
                                                            if self.leftmostRow%7 != 0: new_alloted_R = (new_row == 'R') | (self.alloted_R)
                                                            else: new_alloted_R = np.full(self.alloted_R.shape, False)

                                                            if (new_alloted_R == False).sum() > r*(self.leftmostRow%7 - 1): continue
                                                            if self.leftmostRow%7 == 2:
                                                                min_r_fixed = max(m-a, 0)
                                                                s1 = (new_row[new_alloted_R == False] != 'M').sum()
                                                                if r - min_r_fixed < s1: continue
                                                            
                                                            newarr = self.arr.copy()
                                                            newarr[:, self.leftmostRow - 1] = new_row
                                                            if self.leftmostRow == 1:
                                                                boool, val = check_constraints_b(N, self.D, m, a, e, r, S, newarr)
                                                                if boool:
                                                                    if val > best_val:
                                                                        best_val = val
                                                                        best_arr = newarr
                                                                continue



                                                            else:
                                                                newcountArr = np.full(self.count_arr.shape, None)
                                                                newcountArr[:self.S] = 0
                                                                m1 = np.nonzero((newcountArr!=None) & ((self.curr_row == 'M')|(self.curr_row == 'E')))[0]
                                                                m2 = np.nonzero((newcountArr!=None) & np.logical_not((self.curr_row == 'M')|(self.curr_row == 'E')))[0]
                                                                newcountArr[m1] = self.count_arr[m1]+1
                                                                newcountArr[m2] = self.count_arr[m2]
                                                                
                                                                newtree = Tree2(self.N, self.D, m, a, e, self.S, newarr, self.leftmostRow - 1, new_alloted_R, newcountArr)
                                                                newtree.expand(debug=debug)
                                                            










                        



        
        
    
        

def extracols(N, D, m, a, e):
    assert(D<7)
    r = N - m - a - e
    arr = np.full((N, D), None)
    arr[:, -1] = ['R']*r + ['M']*m + ['A']*a + ['E']*e
    for l in range(D-2, -1, -1):
        r_cnt = r
        a_cnt = a
        none_cnt = N
        prev_none_cnt = none_cnt

        row = arr[:, l+1]
        new_row = np.full(row.shape, None)
        new_row[np.nonzero(row == 'M')[0][:min(m, r)]] = 'R'
        prev_none_cnt = none_cnt; none_cnt = (new_row == None).sum()
        r_cnt -= (prev_none_cnt - none_cnt)
        
        
        new_row[np.nonzero((row == 'M')&(new_row == None))[0][:]] = 'A'
        prev_none_cnt = none_cnt; none_cnt = (new_row == None).sum()
        a_cnt = a - (prev_none_cnt - none_cnt)
        new_row[new_row == None] = ['R']*r_cnt + ['M']*m + ['A']*a_cnt + ['E']*e
        arr[:, l] = new_row
    
    return arr

def allot_next_row(N, D, m, a, e, row1):
    r = N - m - a - e
    r_cnt = r
    a_cnt = a
    none_cnt = N
    prev_none_cnt = none_cnt

    row = row1
    new_row = np.full(row.shape, None)
    new_row[np.nonzero(row == 'M')[0][:min(m, r)]] = 'R'
    prev_none_cnt = none_cnt; none_cnt = (new_row == None).sum()
    r_cnt -= (prev_none_cnt - none_cnt)
    
    
    new_row[np.nonzero((row == 'M')&(new_row == None))[0][:]] = 'A'
    prev_none_cnt = none_cnt; none_cnt = (new_row == None).sum()
    a_cnt = a - (prev_none_cnt - none_cnt)
    new_row[new_row == None] = ['R']*r_cnt + ['M']*m + ['A']*a_cnt + ['E']*e

    return new_row

def get_solution_a(N, D, m, a, e, debug = False):
    no_solution = False
    r = N - m - a - e
    if(D>=7 and 7*r<N): no_solution = True
    elif(a+r<m): no_solution = True
    else:
        r = N - m - a - e
        rem = D%7
        div = D//7
        if(rem == 0):
            extracols_arr = extracols(N, 1, m, a, e)
        else:
            extracols_arr = extracols(N, rem, m, a, e)
        fin_arr = extracols_arr
        
        
        last_row = extracols_arr[:, 0]
        for i in range(div):
            arr = np.full((N, 7), None)
            arr[:, -1] = allot_next_row(N, D, m, a, e, last_row)
            alloted_R = (arr[:, -1] == 'R')
            leftmostRow = 6
            root = Tree(N, 7, m, a, e, arr, leftmostRow, alloted_R)
            if debug and i==0: bool1, res_arr = root.expand(debug = True)
            else: bool1, res_arr = root.expand()
            if bool1:
                bool2 = check_constraints_a(N, 7, m, a, e, r, res_arr)
                if not bool2: 
                    no_solution = True
                    #raise Exception('Error')
                    #sys.exit()
                last_row = res_arr[:, 0]
                if rem == 0 and i == 0:
                    fin_arr = res_arr
                else:
                    fin_arr = np.hstack([res_arr, fin_arr])
            
            else:
                #print();print('False');print(f'N: {N}, D: {D}, m: {m}, a: {a}, e: {e}')
                no_solution = True
    
    if no_solution:
        return False, None
    else:
        return True, fin_arr


def get_new_div_row(N, D, m, a, e, S, curr_row, count_arr):
    r = N -m - a - e
    dict1 = {}
    new_row = np.full(curr_row.shape, None)
    vals, counts = np.unique(count_arr[:S], return_counts=True)
    for i, val in enumerate(vals):
        if val == None: continue
        dict1[val] = counts[i]
    
    noneCnt = (new_row == None).sum()
    a_cnts = a
    e_cnts = e
    m_cnts = m
    r_cnts = r
    
    for k in sorted(dict1.keys())[::-1]:
        m1 = (count_arr == k) & (curr_row == 'M')
        m2 = (count_arr == k) & (curr_row != 'M')

        new_row[np.nonzero(m2 & (new_row == None))[0][:e_cnts]] = 'E'
        prevNoneCnt = noneCnt; noneCnt = (new_row == None).sum();e_cnts -= (prevNoneCnt - noneCnt)

        new_row[np.nonzero(m2 & (new_row == None))[0][:m_cnts]] = 'M'
        prevNoneCnt = noneCnt; noneCnt = (new_row == None).sum();m_cnts -= (prevNoneCnt - noneCnt)

        new_row[np.nonzero(m1 & (new_row == None))[0][:r_cnts]] = 'R'
        prevNoneCnt = noneCnt; noneCnt = (new_row == None).sum();r_cnts -= (prevNoneCnt - noneCnt)

    new_row[np.nonzero((new_row == None) & (curr_row == 'M'))[0][:r_cnts]] = 'R'
    prevNoneCnt = noneCnt; noneCnt = (new_row == None).sum();r_cnts -= (prevNoneCnt - noneCnt)


    new_row[np.nonzero((new_row == None) & (curr_row == 'M'))[0][:a_cnts]] = 'A'
    prevNoneCnt = noneCnt; noneCnt = (new_row == None).sum();a_cnts -= (prevNoneCnt - noneCnt)


    new_row[new_row == None] = ['A']*a_cnts + ['R']*r_cnts + ['M']*m_cnts + ['E']*e_cnts

    return new_row

def get_solution_b(N, D, m, a, e, S, T):
    global div_max_time, calls, best_val, best_arr, print_counter, start_time
    no_solution = False
    fin_arr = None
    r = N - m - a - e
    if(D>=7 and 7*r<N): no_solution = True; print('yo')
    elif(a+r<m): no_solution = True
    else:
        rem = D%7
        div = D//7
        div_max_time = (T/(div + 1))

        fin_arr = np.full((N, rem), None)###
        if rem!=0:
            fin_arr[:, -1] = ['E']*e + ['M']*m + ['A']*a + ['R']*r
            count_arr_temp = np.full(fin_arr[:, -1].shape, None)
            count_arr_temp[:S] = 0
            m1_temp = (count_arr_temp== 0) & ((fin_arr[:, -1] == 'M')|(fin_arr[:, -1] == 'E'))
            count_arr_temp[m1_temp] = 1

            for i in range(rem-2, -1, -1):
                fin_arr[:, i] = get_new_div_row(N, D, m, a, e, S, fin_arr[:, i+1], count_arr_temp)
                count_arr_prev = count_arr_temp
                count_arr_temp = np.full(count_arr_prev.shape, None)
                count_arr_temp[:S] = 0
                m1_temp = (count_arr_temp== 0) & ((count_arr_prev == 'M')|(count_arr_prev == 'E'))
                count_arr_temp[m1_temp] = 1 + count_arr_prev[m1_temp]
                m2_temp = count_arr_temp == 0
                count_arr_temp[m2_temp] = count_arr_prev[m2_temp]
            
            last_row = get_new_div_row(N, D, m, a, e, S, fin_arr[:, 0], count_arr_temp)
        else:
            last_row = ['E']*e + ['M']*m + ['A']*a + ['R']*r
        
        arr = np.full((N, 7), None); arr[:, -1] = last_row
        alloted_R = arr[:, -1] == 'R'
        count_arr = np.full(alloted_R.shape, None)
        count_arr[:S] = 0
        count_arr[(count_arr == 0) & ((arr[:, -1] == 'M')|(arr[:, -1] == 'E'))] = 1

        for d in range(div):
            print('Loop Iteration')
            treeA = Tree2(N, 7, m, a, e, S, arr, 7-1, alloted_R, count_arr)
            #print(N, 7, m, a, e, S, arr, 7-1, alloted_R, count_arr)
            start_time = time.time()
            treeA.expand()
            tree_calls = calls
            tree_val = best_val
            if best_val == -1: 
                b, fin_arr = get_solution_a(N, D, m, a, e)
                no_solution = not b
                break
            else: tree_arr = best_arr
            #print(calls)
            #print(tree_val)
            #print(tree_arr)
            calls = 0
            best_val = -1
            best_arr = None
            print_counter = 0
            if rem == 0 and d==0: fin_arr = tree_arr
            else: fin_arr = np.hstack([tree_arr, fin_arr])

            last_row = get_new_div_row(N, d, m, a, e, S, tree_arr[:, 0], count_arr)
            arr = np.full((N, 7), None); arr[:, -1] = last_row
            alloted_R = arr[:, -1] == 'R'
            count_arr_prev = count_arr
            count_arr = np.full(alloted_R.shape, None)
            count_arr[:S] = 0
            m1 = (count_arr == 0) & ((arr[:, -1] == 'M')|(arr[:, -1] == 'E'))
            count_arr[m1] = 1 + count_arr_prev[m1]
            m2 = count_arr == 0
            count_arr[m2] = count_arr_prev[m2]

        if type(fin_arr) == type(None): no_solution = True
        elif not no_solution:
            boool, val = check_constraints_b(N, D, m, a, e, r, S, fin_arr)
            if boool == False: no_solution = True#; print('False')
    

    return (not no_solution), fin_arr


    
    
    
        
if __name__ == '__main__':
    #here
    params1 = pd.read_csv(sys.argv[1])
    if len(params1.keys()) == 7:
        part = 'b'
    else: part = 'a'

    #N, D, m, a, e, S, T = 28, 34, 13, 9, 2, 9, 60
    #b, sol = get_solution_b(N, D, m, a, e, S, T)
    #print(sol)
    #print(b)

    np_arr = []
    bool_arr = []
    with open("solution.json", 'w') as file:
        for i in range(params1.shape[0]):
            params = params1.iloc[i]
            N = int(params['N'])
            D = int(params['D'])
            m = int(params['m'])
            a = int(params['a'])
            e = int(params['e'])
            if part == 'b':
                S = int(params['S'])
                T = int(params['T'])
                b, sol = get_solution_b(N, D, m, a, e, S, T)
                if b:
                    json.dump(numpy_to_dict(sol), file)
                else:
                    json.dump({}, file)
            else:
                b, sol = get_solution_a(N, D, m, a, e)
                if b:
                    json.dump(numpy_to_dict(sol), file)
                else:
                    json.dump({}, file)
    

    '''for N in range(1, 50):
        for D in range(7,8):
            for m in range(N):
                for a in range(N - m):
                    for e in range(N - m - a):
                        r = N - m - a - e
                        b, sol = get_solution_a(N, D, m, a, e)
                        if b:
                            if(N == 20 and D == 7 and m == 10 and a == 7 and e==0):
                                print(f'N: {N}, D: {D}, m: {m}, a: {a}, e: {e}')
                                print(sol)'''
                


            




    '''arr = np.full((N, D), None)
    arr[:, -1] = ['E']*e + ['M']*m + ['A']*a + ['R']*r 
    alloted_R = arr[:, -1] == 'R'
    count_arr = np.full(alloted_R.shape, None)
    count_arr[:S] = 0
    count_arr[(count_arr == 0) & ((arr[:, -1] == 'M')|(arr[:, -1] == 'E'))] = 1
    tree_new = Tree2(N, D, m, a, e, S, arr, D-1, alloted_R, count_arr)
    
    tree_new.expand()
    print(calls)
    print(best_val)
    print(best_arr)'''




